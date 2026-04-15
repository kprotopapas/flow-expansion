"""
The point of this trainer is that it computes likelihoods of several models and uses them to compute a reward function.
An example application is the Or-operator or Renyi divergence.
"""

from ..models import DiffusionModel
from .adjoint_matching import AdjointMatchingFinetuningTrainer
import torch
from ..likelihood import MultiItoDensODE
from tqdm import tqdm
from .adjoint_matching import LeanAdjointSolver, AMDataset, ConcatDataset, sample_trajectories_ddpm
from ..solvers import TorchDiffEqSolver, PFODE, VPSDE, DDIMSolver
from ..trainers.adjoint_matching import sample_trajectories_ddim, sample_trajectories_ddpm
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
    traj = res[-1]['traj'][0].transpose(1,0) 
    logp = res[0][1] 
    scores = ito_ode.scores
    return traj, logp, scores, ts


def renyi_first_variation_grad(logps, logqs, scorep, scoreq, alpha_renyi, estimate_Z=False):
    """
    Computes grad of first variation of the Renyi divergence $D_\alpha_renyi( p || q)$.
    

    $$
      \nabla_x \partial D_{\alpha_renyi}(p||q) = 
      \frac{\alpha_renyi}{\int_\gX} p(z)^{\alpha_renyi} q(z)^{1-\alpha_renyi} dz
      \Big ( ...  Big)
    $$
    """
    if not estimate_Z:
        Z = 1
    else:
        raise NotImplementedError("Estimate Z not implemented yet")

    # build the log-space coefficients
    coef1 = (alpha_renyi-1)*logps + (1-alpha_renyi)*logqs
    coef2 = (alpha_renyi-1)*logps + (1-alpha_renyi)*logqs

    # clamp them to ±C to avoid huge exponents
    C = 3.0
    coef1 = coef1.clamp(-C, +C)
    coef2 = coef2.clamp(-C, +C)

    # now exponentiate and weight
    term1 = coef1.exp() * scorep
    term2 = coef2.exp() * scoreq
    
    # term1 = ((alpha_renyi-1)*logps + (1-alpha_renyi)*logqs).exp() * scorep  
    # term2 = ((alpha_renyi-1)*logps + (1-alpha_renyi)*logqs).exp() * scoreq

    return Z * (term1 - term2)


class LikelihoodEstTrainer(AdjointMatchingFinetuningTrainer):
    def __init__(self, model: DiffusionModel, 
                 grad_reward,
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
                 clip_grad_norm=None,
                 alpha_renyi=1.0,
                 alpha_div = 1.0,
                 estimate_Z=False):
        
        logger.info("Using first variation of double reverse KL as reward, lambda:{}".format(lmbda))
        grad_reward_fn = lambda x: None
        self.lmbda = lmbda
        self.alpha_renyi = alpha_renyi
        self.estimate_Z = estimate_Z
        self.grad_reward = grad_reward
        self.alpha_div = alpha_div

        super().__init__(model, grad_reward_fn, lr, traj_samples_per_stage, 
                         data_shape, finetune_steps, batch_size, device=device, 
                         base_model=base_model, traj_len=traj_len, 
                         clip_grad_norm=clip_grad_norm, memsave=False)

        self.fine_model = self.fine_model.to(device)
        self.base_model = self.base_model.to(device)
        self.pre_models = pre_models

    def get_grad_reward_fn(self, logps, scores) -> "torch.Tensor[N, D]":
        """Union reward function."""
        logps = logps.to(self.device)
        scores = scores.to(self.device)

        log_p_fine = logps[0,:, None]
        log_p_pre = logps[1,:, None]
        p_fine = logps[0,:, None].exp()
        p_pre = logps[1,:,None].exp()
        scorep = scores[0]
        scoreq = scores[1]

        logger.debug(f"shapes: {p_fine.shape}, {p_pre.shape}, {scorep.shape}, {scoreq.shape}")

        assert len(logps) == len(scores), "logps and scores must have the same length, but got {} and {}".format(len(logps), len(scores))
        assert len(logps)-1 == len(self.pre_models), "logps and pre_models must have the same length, but got {} and {}".format(len(logps)-1, len(self.pre_models))
        assert len(logps) == 2, "logps must have length 2 for just reny divergence, but got {}".format(len(logps))

        rew = renyi_first_variation_grad(log_p_fine, log_p_pre, scorep, scoreq, alpha_renyi=self.alpha_renyi, estimate_Z=self.estimate_Z)

        return lambda _: rew.to('cuda')


    def sample_dataset_stage(self, stages=1, verbose=False):
            """Sample dataset for training based on adjoint ODE and sampled trajectories."""
            
            datasets = []
            # shift model to cuda
            self.fine_model.eval()
            self.base_model.eval()

            dtype = next(self.fine_model.parameters()).dtype
            device = next(self.fine_model.parameters()).device
            logger.info(f"Sampling dataset on {device} with dtype {dtype}")
            with torch.no_grad():
                for _ in tqdm(range(stages), disable=not verbose):
                    x0 = torch.randn(self.traj_samples_per_stage, *self.data_shape).to(self.device).to(dtype)
                    self.fine_model.to(self.device)
                    trajs, logp, scores, ts = sample_trajectories_ito(self.fine_model, self.pre_models, x0=x0, 
                                                              T=self.traj_len, sample_jumps=self.random_jumps) 

                    renyi_grad_fn = self.get_grad_reward_fn(logp, scores) 

                    # compose gradient of the external reward and the gradient of the first variation of the renyi divergence
                    def total_grad_fn(x):
                        g_base  = self.grad_reward(x)      
                        g_renyi = renyi_grad_fn(x)    
                        total_grad = self.lmbda * (g_base - self.alpha_div * g_renyi)

                        # final sanity clamp
                        total_grad = torch.nan_to_num(
                            total_grad,
                            nan=0.0,        # replace any NaN with 0
                            posinf=1e3,     # cap positive Inf to 1e3
                            neginf=-1e3     # cap negative Inf to -1e3
                        )
 
                        return total_grad
                    
                    solver = LeanAdjointSolver(self.base_model, total_grad_fn)

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
