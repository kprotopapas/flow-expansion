import numpy as np
from omegaconf import OmegaConf
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, ConcatDataset

from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment


class LeanAdjointSolverFlow:
    """Lean adjoint matching solver (Domingo-Enrich et al. 2024) using DDProtocol types."""

    def __init__(
        self,
        base_model: BaseModel,
        grad_reward_fn: Callable,
        grad_f_k_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = base_model
        self.scheduler = base_model.scheduler
        self.grad_reward_fn = grad_reward_fn
        self.grad_f_k_fn = grad_f_k_fn
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def step(self, adj, x_t, t: torch.Tensor, dt: torch.Tensor):
        """Single backward step of the lean adjoint ODE.

        adj, x_t: DDMixin (D) — adjoint and forward state
        t: Tensor shape (batch,) — current timestep
        dt: scalar Tensor — step size (positive)
        """
        adj_t = adj.detach()

        with torch.enable_grad():
            x_t_grad = x_t.detach().requires_grad(True)
            v_pred = self.model.forward(x_t_grad, t)

            alpha = self.scheduler.alpha(x_t_grad, t)
            alpha_dot = self.scheduler.alpha_dot(x_t_grad, t)

            eps_pred = 2 * v_pred - alpha_dot / (alpha + dt) * x_t_grad
            g_term = (adj_t * eps_pred).aggregate("sum").sum()
            v = x_t_grad.gradient(g_term)

        adj_tmh = adj_t + dt * v
        if self.grad_f_k_fn is not None:
            grad_f_k = self.grad_f_k_fn(x_t, t + dt)
            adj_tmh = adj_t + dt * v - dt * grad_f_k

        return adj_tmh.detach(), v_pred.detach()

    def solve(self, trajectories: list, ts: torch.Tensor) -> dict:
        """Backward adjoint pass over a forward trajectory.

        trajectories: list[D] of length T
        ts: Tensor shape (T,) — forward timesteps

        Returns dict with 't', 'traj_x', 'traj_adj', 'traj_v_pred' in forward-time order.
        """
        T = ts.shape[0]
        assert T == len(trajectories)
        dt = ts[1] - ts[0]
        ts_rev = ts.flip(0)

        n = len(trajectories[0])
        x_1 = trajectories[-1]
        adj = -self.grad_reward_fn(x_1)

        trajs_adj = []
        traj_v_pred = []

        for i in range(1, T):
            t_batch = ts_rev[i].unsqueeze(0).expand(n).to(self.device)
            x_t = trajectories[T - i - 1]
            adj, v_pred = self.step(adj=adj, x_t=x_t, t=t_batch, dt=dt)
            trajs_adj.append(adj.detach())
            traj_v_pred.append(v_pred.detach())

        return {
            't': ts[:-1],
            'traj_x': trajectories[:-1],
            'traj_adj': list(reversed(trajs_adj)),
            'traj_v_pred': list(reversed(traj_v_pred)),
        }


class AMDataset(Dataset):
    def __init__(self, solver_info: dict):
        self.t = solver_info['t']
        self.traj_x = solver_info['traj_x']
        self.traj_adj = solver_info['traj_adj']
        self.traj_v_base = solver_info['traj_v_pred']
        self.traj_sigma = solver_info['traj_sigma']
        self.T = self.t.size(0)
        self.bs = 1

    def __len__(self):
        return self.bs

    def __getitem__(self, idx):
        return {
            'ts': self.t,
            'traj_x': self.traj_x,
            'traj_adj': self.traj_adj,
            'traj_v_base': self.traj_v_base,
            'traj_sigma': self.traj_sigma,
        }


def create_timestep_subset(total_steps, final_percent=0.25, sample_percent=0.25):
    """Create a subset of time-steps for efficient computation (Appendix G2)."""
    final_steps_count = int(total_steps * final_percent)
    sample_steps_count = int(total_steps * sample_percent)
    final_samples = np.arange(final_steps_count)
    remaining_steps = np.setdiff1d(np.arange(total_steps), final_samples)
    additional_samples = np.random.choice(remaining_steps, size=sample_steps_count, replace=False)
    return np.sort(np.concatenate([final_samples, additional_samples]))


def adj_matching_loss(v_base, v_fine, adj, sigma) -> torch.Tensor:
    """Adjoint matching loss for a single timestep batch (D-typed inputs)."""
    diff = v_fine - v_base
    term_diff = (2 / sigma) * diff
    term_adj = sigma * adj
    term_difference = term_diff - term_adj
    return (term_difference ** 2).aggregate("sum").mean()


class AMTrainerFlow:
    def __init__(
        self,
        config: OmegaConf,
        env: Environment,
        fine_model: BaseModel,
        base_model: BaseModel,
        grad_reward_fn: Callable,
        grad_f_k_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        self.config = config
        self.sampling_config = config.sampling
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self.clip_grad_norm = config.get("clip_grad_norm", 1e5)
        self.clip_loss = config.get("clip_loss", 0.5)

        self.env = env
        self.fine_model = fine_model
        self.base_model = base_model
        self.fine_model.to(self.device)
        self.base_model.to(self.device)

        self.grad_reward_fn = grad_reward_fn
        self.grad_f_k_fn = grad_f_k_fn

        self.configure_optimizers()

    def configure_optimizers(self):
        if hasattr(self, 'optimizer'):
            del self.optimizer
        self.optimizer = torch.optim.Adam(self.fine_model.parameters(), lr=self.config.lr)

    def get_model(self):
        return self.fine_model

    @torch.no_grad()
    def sample_trajectories(self):
        """Sample trajectories from the fine model using env."""
        original_base = self.env.base_model
        self.env.base_model = self.fine_model
        env_sample = self.env.sample(self.config.batch_size, pbar=False)
        self.env.base_model = original_base
        return env_sample

    def generate_dataset(self):
        """Build training dataset by running the adjoint ODE over sampled trajectories."""
        datasets = []
        self.fine_model.eval()
        self.base_model.eval()

        solver = LeanAdjointSolverFlow(
            self.base_model, self.grad_reward_fn, self.grad_f_k_fn, self.device
        )

        iterations = max(1, self.sampling_config.num_samples // self.config.batch_size)
        for _ in range(iterations):
            env_sample = self.sample_trajectories()

            traj = [x.to(self.device) for x in env_sample.trajectory]
            ts = env_sample.timesteps.to(self.device)

            solver_info = solver.solve(trajectories=traj, ts=ts)
            solver_info['traj_sigma'] = [d.to("cpu") for d in env_sample.diffusions]

            datasets.append(AMDataset(solver_info=solver_info))

        if not datasets:
            return None
        return ConcatDataset(datasets)

    def train_step(self, sample: dict) -> torch.Tensor:
        ts = sample['ts'].to(self.device)
        traj_x = [x.to(self.device) for x in sample['traj_x']]
        traj_adj = [a.to(self.device) for a in sample['traj_adj']]
        traj_v_base = [v.to(self.device) for v in sample['traj_v_base']]
        traj_sigma = [s.to(self.device) for s in sample['traj_sigma']]

        idxs = create_timestep_subset(ts.shape[0])

        losses = []
        for idx in idxs:
            n = len(traj_x[idx])
            t = ts[idx].unsqueeze(0).expand(n)
            v_fine_t = self.fine_model.forward(traj_x[idx], t)
            loss_t = adj_matching_loss(traj_v_base[idx], v_fine_t, traj_adj[idx], traj_sigma[idx])
            losses.append(loss_t)

        if not losses:
            return torch.tensor(float("inf"), device=self.device)

        loss = torch.stack(losses).mean()
        if loss.isnan().any():
            return torch.tensor(float("inf"), device=self.device)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)

        if self.clip_loss > 0.0:
            loss = torch.clamp(loss, min=0.0, max=self.clip_loss)
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.fine_model.parameters(), self.clip_grad_norm)

        self.optimizer.step()
        return loss

    def finetune(self, dataset, steps=None, debug=False):
        c = 0
        losses = []
        self.fine_model.to(self.device)
        self.fine_model.train()
        self.optimizer.zero_grad()

        idxs = np.random.permutation(len(dataset))
        if steps is not None:
            idxs = idxs[:steps]

        for idx in idxs:
            sample = dataset[int(idx)]
            loss = self.train_step(sample).item()
            losses.append(loss)
            c += 1

        del dataset
        return losses if debug else sum(losses) / c
