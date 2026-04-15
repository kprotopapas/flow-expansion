from omegaconf import OmegaConf
from genexp.models import FlowModel
from genexp.sampling import Sampler
from genexp.trainers.adjoint_matching import AMTrainerFlow
from typing import Callable, Optional

import torch
from torch import Tensor
import copy


class FlowExpansionTrainer(AMTrainerFlow):
    def __init__(self,
                 config: OmegaConf,
                 model: FlowModel,
                 base_model: FlowModel,
                 device: torch.device = None,
                 verbose: bool = False,
                 grad_constraint: Optional[Callable] = None,
                 sampler: Optional[Sampler] = None):
            
        self.gamma: float = config.get('gamma', 1.)
        self.eta: float = config.get('eta', 1.)
        self.beta: float = config.get('beta', 0.)
        self.epsilon = torch.tensor(config.epsilon).to(device)

        self.grad_constraint = grad_constraint
        self.traj: bool = config.traj
        self.base_base_model = copy.deepcopy(base_model)

        self.lmbda_schedule: str = config.get('lmbda', 'const')
        
        if self.lmbda_schedule == 'variance':
            self.lmbda = model.interpolant_scheduler.memoryless_sigma_t
        elif self.lmbda_schedule == 'cosine':
            raise NotImplementedError()
        elif self.lmbda_schedule == 'const':
            self.lmbda = lambda _: torch.tensor(1.).to(device)
        else:
            raise ValueError()
        
        self.combined_score = lambda s, t: base_model.score_func(s, t) - self.beta * self.base_base_model.score_func(s, t)


        def grad_reward_fn(x: Tensor) -> Tensor:
            return -self.gamma * self.lmbda(1. - self.epsilon) * self.combined_score(x, 1. - self.epsilon)
    

        def grad_f_k_fn(x: Tensor, t: Tensor) -> Tensor:
            return -self.gamma * self.lmbda(t) * self.combined_score(x, torch.min(t, 1. - self.epsilon))
    

        self.grad_reward_fn = grad_reward_fn
        self.grad_f_k_fn = grad_f_k_fn

        self.grad_constraint = grad_constraint

        traj_fn = grad_f_k_fn if self.traj else None

        super().__init__(config.adjoint_matching, model, base_model, grad_reward_fn, traj_fn, device, verbose, sampler)


    def expand(self):
        lmbda = self.lmbda_schedule
        if lmbda == 'variance':
            self.lmbda = model.interpolant_scheduler.memoryless_sigma_t
        elif lmbda == 'cosine':
            raise NotImplementedError()
        elif lmbda == 'const':
            self.lmbda = lambda t: 1.
        else:
            raise ValueError()
        
        self.combined_score = lambda s, t: self.base_model.score_func(s, t) - self.beta * self.base_base_model.score_func(s, t)

        def grad_reward_fn(x: Tensor) -> Tensor:
            return -self.gamma * self.lmbda(1. - self.epsilon) * self.combined_score(x, 1. - self.epsilon)
    

        def grad_f_k_fn(x: Tensor, t: Tensor) -> Tensor:
            return -self.gamma * self.lmbda(t) * self.combined_score(x, torch.min(t, 1. - self.epsilon))
    
        self.grad_reward_fn = grad_reward_fn
        self.grad_f_k_fn = grad_f_k_fn if self.traj else None
    

    def project(self):
        self.grad_reward_fn = lambda x: self.eta * self.grad_constraint(x)
        self.grad_f_k_fn = None


    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())
