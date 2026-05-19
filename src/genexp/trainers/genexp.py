from omegaconf import DictConfig
from diffusiongym.base_models import BaseModel
from diffusiongym.environments import Environment
from diffusiongym.types import D
from genexp.trainers.adjoint_matching import AMTrainerFlow
from typing import Callable, Optional
from tqdm import tqdm

import torch
import copy


def _score_func(model: BaseModel[D], x: D, t: torch.Tensor) -> D:
    """Compute the score function ∇log p_t(x) from a velocity-predicting model."""
    if model.output_type == "score":
        return model.forward(x, t)
    
    elif model.output_type == "velocity":
        v = model.forward(x, t)
        scheduler = model.scheduler
        kappa = scheduler.kappa(x, t)   # α_dot / α
        eta = scheduler.eta(x, t)       # β(κβ - β_dot)
        return (v - kappa * x) / eta

    elif model.output_type == "endpoint":
        x_1 = model.forward(x, t)
        scheduler = model.scheduler
        alpha = scheduler.alpha(x, t)
        beta = scheduler.beta(x, t)
        return (alpha * x_1 - x) / (beta ** 2)

    elif model.output_type == "epsilon":
        eps = model.forward(x, t)
        beta = model.scheduler.beta(x, t)
        return -eps / beta

    raise ValueError("Incorrectly specified base model")


class FlowExpansionTrainer(AMTrainerFlow):
    def __init__(
        self,
        config: DictConfig,
        env: Environment,
        model: BaseModel,
        base_model: BaseModel,
        device: Optional[torch.device] = None,
        verbose: bool = False,
        grad_constraint: Optional[Callable] = None,
    ):
        self.gamma: float = config.get('gamma', 1.)
        self.eta_coeff: float = config.get('eta', 1.)
        self.beta: float = config.get('beta', 0.)
        self.epsilon = torch.tensor(config.epsilon, dtype=torch.float32)
        if device is not None:
            self.epsilon = self.epsilon.to(device)

        self.grad_constraint: Optional[Callable] = grad_constraint
        self.traj: bool = config.traj
        self.base_base_model = copy.deepcopy(base_model)

        self.lmbda_schedule: str = config.get('lmbda', 'const')

        grad_reward_fn, grad_f_k_fn = self._make_fns(base_model, self.base_base_model)
        traj_fn = grad_f_k_fn if self.traj else None

        super().__init__(
            config.adjoint_matching, env, model, base_model,
            grad_reward_fn, traj_fn, device, verbose,
        )

    def _lmbda(self, model, x, t):
        if self.lmbda_schedule == 'variance':
            return model.scheduler.sigma(x, t)
        return 1.0

    def _combined_score(self, base_model, base_base_model, x, t):
        return _score_func(base_model, x, t) - self.beta * _score_func(base_base_model, x, t)

    def _make_fns(self, base_model, base_base_model):
        eps = float(self.epsilon)
        gamma = self.gamma

        def grad_reward_fn(x):
            t = torch.full((len(x),), 1.0 - eps, device=x.device)
            score = self._combined_score(base_model, base_base_model, x, t)
            return -gamma * self._lmbda(base_model, x, t) * score

        def grad_f_k_fn(x, t: torch.Tensor):
            t_clip = t.clamp(max=1.0 - eps)
            score = self._combined_score(base_model, base_base_model, x, t_clip)
            return -gamma * self._lmbda(base_model, x, t_clip) * score

        return grad_reward_fn, grad_f_k_fn

    def expand(self):
        """Update reward functions to use the current base model."""
        grad_reward_fn, grad_f_k_fn = self._make_fns(self.base_model, self.base_base_model)
        self.grad_reward_fn = grad_reward_fn
        self.grad_f_k_fn = grad_f_k_fn if self.traj else None

    def project(self):
        """Switch to the constraint-gradient reward (projection step)."""
        if self.grad_constraint is None:
            raise ValueError('Projection step with no grad_constraint set')
        
        grad_constraint: Callable = self.grad_constraint
        eta = self.eta_coeff
        self.grad_reward_fn = lambda x: eta * grad_constraint(x)
        self.grad_f_k_fn = None

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def fit(self, num_iterations: int, pbar: bool = False) -> list[float]:
        """Run the full expand-project mirror-descent loop.

        Each iteration:
          1. expand()  — AM fine-tuning toward higher reward
          2. project() — AM fine-tuning back toward the constraint set
          3. update_base_model()

        The number of AM rounds per step and gradient steps per round are
        read from config.adjoint_matching.num_iterations and
        config.adjoint_matching.finetune_steps respectively.

        Returns a flat list of per-AM-round losses (expand losses first,
        then project losses, for each mirror-descent iteration).
        """
        am_iters = self.config.get('num_iterations', 1)
        finetune_steps = self.config.get('finetune_steps', None)
        losses = []
        
        it = tqdm(range(num_iterations)) if pbar else range(num_iterations)

        for _ in it:
            self.expand()
            for _ in range(am_iters):
                dataset = self.generate_dataset()
                losses.append(self.finetune(dataset, steps=finetune_steps))

            if self.grad_constraint is not None and self.eta_coeff > 0:
                self.project()
                for _ in range(am_iters):
                    dataset = self.generate_dataset()
                    losses.append(self.finetune(dataset, steps=finetune_steps))

            self.update_base_model()
        

        return losses
