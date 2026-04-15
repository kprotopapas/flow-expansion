from ..models import DiffusionModel
from .adjoint_matching import AdjointMatchingFinetuningTrainer
import torch
from ..likelihood import MultiItoDensODE
from tqdm import tqdm
from .adjoint_matching import LeanAdjointSolver, AMDataset, ConcatDataset, sample_trajectories_ddpm
from ..solvers import TorchDiffEqSolver, PFODE, VPSDE, DDIMSolver
from .adjoint_matching import sample_trajectories_ddim, sample_trajectories_ddpm
from ..likelihood import prior_likelihood

import logging
logger = logging.getLogger(__name__)


def sample_trajectories_ito(model: DiffusionModel, pre_models: torch.nn.ModuleList,x0, T, avoid_inf=0.0, sample_jumps=True):
    """Sample trajectories from the basqqqqed on probflow ODE, also estimate likelihoods of several models."""
    device = next(model.parameters()).device
    x0 = x0.to(device)
    ito_ode = MultiItoDensODE(model_sampling=model, models=pre_models, sde=model.sde, sign=1)
    ts = torch.linspace(1,0, T, device=device)
    plik = prior_likelihood(x0, 1.0)
    plik = torch.stack([plik] + [plik for _ in range(len(ito_ode.models))])
    x0 = (x0, plik)
    solver = TorchDiffEqSolver(ito_ode)
    res = solver.solve(x0, t=ts, method='euler', atol=1e-5)    
    traj = res[-1]['traj'][0].transpose(1,0) # (n, T, d)
    logp = res[0][1] # (m, n), at [0] we have the logp of the sampling model
    scores = ito_ode.scores
    return traj, logp, scores, ts


class OrOperatorTrainer(AdjointMatchingFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 lr, 
                 traj_samples_per_stage, 
                 data_shape, 
                 pre_models,
                 finetune_steps=100, 
                 batch_size=32, 
                 device='cuda',
                 base_model=None,
                 traj_len=100,
                 lmbda=1.0,
                 clip_grad_norm=None):
        
        print("Using first variation of double reverse KL as reward, lambda:", lmbda)
        grad_reward_fn = lambda x: None
        self.lmbda = lmbda

        super().__init__(model, grad_reward_fn, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, 
                         clip_grad_norm=clip_grad_norm, memsave=False)

        self.fine_model = self.fine_model.to(device)
        self.base_model = self.base_model.to(device)
        self.pre_models = pre_models

    def get_grad_reward_fn(self, logps, scores):
        """Union reward function."""
        logps = logps.to(self.device)
        scores = scores.to(self.device)

        p_fine = logps[0,None,:, None].exp()
        p_pre = logps[1:,:,None].exp()
        score_fine = scores[0,None]
        score_pre = scores[1:]
        rew = torch.mean((p_pre/p_fine) * (score_fine - score_pre),dim=0)
        return lambda _: rew.to('cuda')


    def sample_dataset_stage(self, stages=1, verbose=False):
            """Sample dataset for training based on adjoint ODE and sampled trajectories."""
            
            datasets = []
            # shift model to cuda
            self.fine_model.eval()
            self.base_model.eval()

            # check model devices
            # for model in self.pre_models:
            #     print('pre model device', next(model.parameters()).device)
            # print('fine model device', next(self.fine_model.parameters()).device)
            # print('base model device', next(self.base_model.parameters()).device)


            dtype = next(self.fine_model.parameters()).dtype
            device = next(self.fine_model.parameters()).device
            logger.info(f"Sampling dataset on {device} with dtype {dtype}")
            with torch.no_grad():
                for _ in tqdm(range(stages), disable=not verbose):
                    x0 = torch.randn(self.traj_samples_per_stage, *self.data_shape).to(self.device).to(dtype)
                    self.fine_model.to(self.device)
                    trajs, logp, scores, ts = sample_trajectories_ito(self.fine_model, self.pre_models, x0=x0, 
                                                              T=self.traj_len, sample_jumps=self.random_jumps) 
                    # create reward function
                    grad_reward_fn = self.get_grad_reward_fn(logp, scores)
                    solver = LeanAdjointSolver(self.base_model, grad_reward_fn)

                    if self.memsave:
                        logger.debug('shifting to cpu')
                        self.fine_model.to('cpu')
                    _, solver_info = solver.solve(trajs.to('cuda'), ts=ts.flip(0).to('cuda'))
                    if self.memsave:
                        logger.debug('shifting to cpu')
                        self.base_model.to('cpu')
                    dataset = AMDataset(solver_info=solver_info)
                    datasets.append(dataset)
            dataset = ConcatDataset(datasets)
            return dataset

    def update_reward(self):
        self.grad_reward_fn = lambda x: -self.base_model.score_func(x, torch.tensor(0.0, device=x.device).float().detach())*self.lmbda

    def update_base_model(self):
        self.base_model.load_state_dict(self.fine_model.state_dict())

    def set_lambda(self, lmbda):
        self.lmbda = lmbda
        self.update_reward()
