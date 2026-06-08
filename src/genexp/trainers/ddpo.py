import numpy as np
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, ConcatDataset

from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment
from diffusiongym.environments.base import Sample
from diffusiongym.schedulers import Scheduler
from diffusiongym.types import D

from genexp.trainers.adjoint_matching import create_timestep_subset


def _policy_drift(model: BaseModel[D], x: D, t: torch.Tensor, **kwargs) -> D:
    """Compute the SDE drift used by each environment type from model output.

    Mirrors the `a * x + b * action` formulas in each Environment.drift().
    No control correction term — only the policy-dependent mean.
    """
    output = model.forward(x, t, **kwargs)
    scheduler: Scheduler = model.scheduler

    if model.output_type == "epsilon":
        kappa = scheduler.kappa(x, t)
        eta = scheduler.eta(x, t)
        sigma = scheduler.sigma(x, t)
        beta = scheduler.beta(x, t)
        b = -(0.5 * sigma * sigma + eta) / beta
        return kappa * x + b * output

    elif model.output_type == "velocity":
        kappa = scheduler.kappa(x, t)
        eta = scheduler.eta(x, t)
        sigma = scheduler.sigma(x, t)
        sigma_div_eta = sigma * sigma / (2 * eta)
        a = -sigma_div_eta * kappa
        b = sigma_div_eta + 1
        return a * x + b * output

    elif model.output_type == "score":
        kappa = scheduler.kappa(x, t)
        eta = scheduler.eta(x, t)
        sigma = scheduler.sigma(x, t)
        b = 0.5 * sigma * sigma + eta
        return kappa * x + b * output

    elif model.output_type == "endpoint":
        kappa = scheduler.kappa(x, t)
        eta = scheduler.eta(x, t)
        sigma = scheduler.sigma(x, t)
        alpha = scheduler.alpha(x, t)
        beta = scheduler.beta(x, t)
        sigma_eta = 0.5 * sigma * sigma + eta
        a = kappa - sigma_eta / (beta * beta)
        b = sigma_eta * alpha / (beta * beta)
        return a * x + b * output

    raise ValueError(f"Unknown output_type: {model.output_type}")


def _step_log_prob(
    x_curr: D,
    x_next: D,
    drift: D,
    diffusion: D,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Log probability of the SDE transition x_curr -> x_next under a given drift.

    x_{t+dt} = x_t + dt * drift + sqrt(dt) * sigma * eps,  eps ~ N(0, I)

    Returns shape (batch,).
    """
    mean = x_curr + dt * drift
    std = torch.sqrt(dt) * diffusion
    eps = (x_next - mean) / (std + 1e-8)
    return -0.5 * (eps**2).aggregate("sum")


@dataclass
class DDPOSample:
    trajectory: list  # list[D], length T+1
    timesteps: torch.Tensor
    noises: list  # list[D], length T
    diffusions: list  # list[D], length T
    advantages: torch.Tensor
    kwargs: dict[str, Any] = field(default_factory=dict)


class DDPODataset(Dataset):
    def __init__(self, env_sample: Sample, advantages: torch.Tensor):
        self.sample = DDPOSample(
            trajectory=env_sample.trajectory,
            timesteps=env_sample.timesteps,
            noises=env_sample.noises,
            diffusions=env_sample.diffusions,
            advantages=advantages,
            kwargs=env_sample.kwargs,
        )
        self.T = len(self.sample.noises)
        self.bs = 1

    def __len__(self):
        return self.bs

    def __getitem__(self, index) -> DDPOSample:
        return self.sample


class DDPOTrainer:
    """DDPO trainer (Black et al. 2023) using diffusiongym types.

    Fine-tunes a diffusion model via PPO policy gradient: trajectories are
    sampled with the current policy, rewards normalised to advantages, and a
    clipped PPO loss is computed over a random subset of timesteps.

    Config keys
    -----------
    batch_size          : int   — samples per env.sample() call
    lr                  : float — Adam learning rate
    clip_range          : float — PPO ε (default 0.2)
    adv_clip_max        : float — advantage clamp magnitude (default 10)
    clip_grad_norm      : float — gradient-norm clip (default 1.0)
    sampling.num_samples: int   — total trajectories per outer iteration
    num_inner_epochs    : int   — PPO inner epochs per dataset (default 1)
    timestep_fraction   : float — fraction of timesteps to train on (default 1.0)
    """

    def __init__(
        self,
        config: DictConfig,
        env: Environment,
        fine_model: BaseModel,
        device: Optional[torch.device] = None,
        verbose: bool = False,
        use_valids: bool = False,
    ):
        self.config = config
        self.sampling_config = config.sampling
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self.use_valids = use_valids
        self.clip_range: float = config.get("clip_range", 0.2)
        self.adv_clip_max: float = config.get("adv_clip_max", 10.0)
        self.clip_grad_norm: float = config.get("clip_grad_norm", 1.0)
        self.num_inner_epochs: int = config.get("num_inner_epochs", 1)
        self.timestep_fraction: float = config.get("timestep_fraction", 1.0)

        self.env = env
        self.fine_model = fine_model
        self.fine_model.to(self.device)

        self.configure_optimizers()

    def configure_optimizers(self):
        if hasattr(self, "optimizer"):
            del self.optimizer
        self.optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=self.config.lr)

    def get_model(self) -> BaseModel:
        return self.fine_model

    @torch.no_grad()
    def sample_trajectories(self) -> Sample:
        """Sample one batch of trajectories using the fine model as policy."""
        original_policy = self.env._policy
        original_base = self.env.base_model
        self.env.policy = self.fine_model
        self.env.base_model = self.fine_model
        try:
            env_sample = self.env.sample(self.config.batch_size, pbar=False)
        finally:
            self.env._policy = original_policy
            self.env.base_model = original_base
        return env_sample

    def generate_dataset(self) -> Optional[ConcatDataset]:
        """Collect trajectories, compute global advantages, build training dataset."""
        self.fine_model.eval()

        iterations = max(1, self.sampling_config.num_samples // self.config.batch_size)
        all_samples: list[Sample] = []
        for _ in range(iterations):
            all_samples.append(self.sample_trajectories())

        all_rewards = torch.cat([s.valids if self.use_valids else s.rewards for s in all_samples])
        advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

        datasets = []
        offset = 0
        for sample in all_samples:
            n = len(sample)
            adv = advantages[offset : offset + n]
            offset += n
            datasets.append(DDPODataset(sample, adv))

        return ConcatDataset(datasets) if datasets else None

    def train_step(self, sample: DDPOSample) -> torch.Tensor:
        trajectory = sample.trajectory
        timesteps = sample.timesteps
        noises = sample.noises
        diffusions = sample.diffusions
        advantages = sample.advantages.to(self.device)
        kwargs = sample.kwargs

        T = len(noises)
        idxs = (
            create_timestep_subset(
                T,
                final_percent=0.25,
                sample_percent=max(0.0, self.timestep_fraction - 0.25),
            )
            if self.timestep_fraction < 1.0
            else np.arange(T)
        )

        adv_clipped = torch.clamp(advantages, -self.adv_clip_max, self.adv_clip_max)

        losses = []
        for idx in idxs:
            x_curr = trajectory[idx].to(self.device).detach()
            x_next = trajectory[idx + 1].to(self.device).detach()

            dt = (timesteps[idx + 1] - timesteps[idx]).to(self.device)
            sigma = diffusions[idx].to(self.device)
            eps_old = noises[idx].to(self.device)

            n = len(x_curr)
            t_batch = timesteps[idx].unsqueeze(0).expand(n).to(self.device)
            step_kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

            # Differentiable drift from the current fine model
            drift_new = _policy_drift(self.fine_model, x_curr, t_batch, **step_kwargs)

            # Log prob of x_next under current policy
            log_prob_new = _step_log_prob(x_curr, x_next, drift_new, sigma, dt)

            # Log prob of x_next under old (sampling) policy
            with torch.no_grad():
                # Old noise was sampled from N(0, I), so log_prob_old = -0.5 * ||eps_old||^2
                log_prob_old = -0.5 * (eps_old**2).aggregate("sum")

            ratio = torch.exp(log_prob_new - log_prob_old)

            unclipped = -adv_clipped * ratio
            clipped = -adv_clipped * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            loss_t = torch.mean(torch.maximum(unclipped, clipped))
            losses.append(loss_t)

        if not losses:
            return torch.tensor(float("inf"), device=self.device)

        loss = torch.stack(losses).mean()
        if loss.isnan():
            return torch.tensor(float("inf"), device=self.device)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.fine_model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        return loss.detach()

    def finetune(
        self,
        dataset: ConcatDataset,
        steps: Optional[int] = None,
        debug: bool = False,
    ):
        """Run PPO inner epochs over the collected dataset.

        Returns per-step losses if debug=True, else mean loss.
        """
        self.fine_model.to(self.device)
        self.fine_model.train()

        losses = []
        for _ in range(self.num_inner_epochs):
            idxs = np.random.permutation(len(dataset))
            if steps is not None:
                idxs = idxs[:steps]

            for idx in idxs:
                sample = dataset[int(idx)]
                loss = self.train_step(sample).item()
                losses.append(loss)

        if not losses:
            return [] if debug else float("inf")
        return losses if debug else sum(losses) / len(losses)

    def fit(self, num_iterations: int, pbar: bool = False) -> list[float]:
        """Run the DDPO outer loop: sample → compute advantages → PPO update.

        Returns a flat list of mean losses, one per outer iteration.
        """
        from tqdm import tqdm

        losses = []
        it = tqdm(range(num_iterations)) if pbar else range(num_iterations)
        for _ in it:
            dataset = self.generate_dataset()
            if dataset is not None:
                losses.append(self.finetune(dataset))
        return losses
