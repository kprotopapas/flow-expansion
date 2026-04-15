import torch
from .solvers import ODE, VPSDE, TorchDiffEqSolver
from .models import DiffusionModel
import numpy as np
from matplotlib import pyplot as plt


# computation prior likelihood 
def prior_likelihood(z, sigma):
  """The likelihood of a Gaussian distribution with mean zero and standard deviation sigma."""
  shape = z.shape
  N = np.prod(shape[1:]).astype(np.float32)
  return -N / 2. * np.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,)) / (2 * sigma**2)

def skilling_hutchinson_divergence(x, f, eps=None, dim=[1]):
    """Compute the divergence with Skilling-Hutchinson for f(x)."""
    if eps is None:
        eps = torch.randn_like(x)
    with torch.enable_grad():
      out = torch.sum(f * eps)
      grad_x_f = torch.autograd.grad(out, x, retain_graph=True)[0]
    return torch.sum(grad_x_f * eps, dim=dim)    

class SkillingHutchinsonDivergence(torch.nn.Module):
    def __init__(self, mc_samples=1, dim=[1], batched=False):
       super().__init__()
       self.mc_samples = mc_samples
       self.dim = dim
       self.batched = batched
    
    def forward(self, x, f, eps=None):
        # ---- Hutchinson divergence with mc_samples ----
        div_est = torch.zeros(x.size(0), device=x.device)  # shape (batch,)
        with torch.enable_grad():
          if not self.batched:
            for _ in range(self.mc_samples):
                eps = torch.randn_like(x)      
                out = (f * eps).sum()         
                grad_x_f = torch.autograd.grad(out, x, retain_graph=True if _ < self.mc_samples - 1 else False)[0]
                div_est += (grad_x_f * eps).sum(dim=1)
          else:
              eps = torch.randn(self.mc_samples, *x.size(), device=x.device)
              out = (f[None] * eps).sum()
              grad_x_f = torch.autograd.grad(out, x, retain_graph=False)[0]
              div_est = (grad_x_f[None] * eps).sum(dim=0).sum(dim=1)

        # Average over mc_samples
        div_est = div_est / self.mc_samples
        return div_est



class FullDivergence(torch.nn.Module):
   def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
    

   def forward(self, x, f, eps=None):
      with torch.enable_grad():
        # compute jacobian of f wrt. x
        trace = torch.zeros(x.size(0), device=x.device)
        for i in range(f.size(1)):
            out = f[:, i].sum()
            grad_x_f = torch.autograd.grad(out, x, retain_graph=True if i < f.size(1) - 1 else False)[0]
            trace += grad_x_f[:, i]
        return trace
      

class LikPFODE(ODE):
    """Compute likelihood by divergence of PF-ODE velocity."""

    def __init__(self, model: DiffusionModel, sde: VPSDE, divergence_estimator, sign=1):
      """The convention is that sign=1 when we are computing likelihood while generating (t=1 to t=0)."""
      self.model = model
      self.sde = sde
      self.divergence_estimator = divergence_estimator
      self.dtype = next(model.parameters()).dtype
      self.sign = sign

    def f(self, x, t):
       x, log_det = x
       x = x.to(self.dtype)
       log_det = log_det.to(self.dtype)
       x_ = x.detach().requires_grad_(True)
       v = self.sde.pf_ode_vel(x_, t, self.model)
       log_det = self.divergence_estimator(x_, v)
       return v, self.sign*(-log_det)

class ItoDensODE(ODE):
    """Compute likelihood by Ito Density Estimator, roughly same complexity as sampling with PF-ODE."""
    def __init__(self, model: DiffusionModel, sde: VPSDE, sign=1):
      """The convention is that sign=1 when we are computing likelihood while generating (t=1 to t=0)."""

      self.model = model
      self.sde = sde
      self.dtype = next(model.parameters()).dtype
      self.sign = sign


    def _vel_and_vel_logdet(self, x, t_expand, score, sampling_score=None):
        if sampling_score is None:
            sampling_score = score # evaluating likelihood under the sampling model
        drift, diffusion = self.sde.sde(x,t_expand)
        b_t = self.sde.beta_t(t_expand)
        drift = -0.5 * b_t * x
        # beta_t = self.model.sde.beta_t(t)
        drift, diffusion = self.sde.sde(x,t_expand)
        b_t = self.sde.beta_t(t_expand)
        # 4) Convert eps_pred => score factor: (+0.5 * beta(t) * eps_pred / sigma_t)
        drift = -0.5 * b_t * x  # shape (batch, dim)
        score_term = -0.5 * b_t * score 
        dxdt = drift + score_term

        dxt_dot_score = (dxdt * score).sum(dim=-1)
        div_drift =  ((-b_t*.5)*x.shape[-1]).flatten()  # TODO this is specific for VP-SDE

        assert div_drift.shape == dxt_dot_score.shape

        vel_logdet2 = div_drift + ((drift - b_t/2.* score)*score).sum(dim=1)
        vel_logdet1 = dxt_dot_score
        v_logdet = vel_logdet1+vel_logdet2
        return dxdt, self.sign*(-v_logdet), score

    def _vel_logdet(self, x, t_expand, score):
        drift, diffusion = self.sde.sde(x,t_expand)
        b_t = self.sde.beta_t(t_expand)
        drift = -0.5 * b_t * x
        # beta_t = self.model.sde.beta_t(t)
        b_t = self.sde.beta_t(t_expand)
        # 4) Convert eps_pred => score factor: (+0.5 * beta(t) * eps_pred / sigma_t)
        drift = -0.5 * b_t * x  # shape (batch, dim)
        score_term = -0.5 * b_t * score 
        dxdt = drift + score_term
        dxt_dot_score = (dxdt * score).sum(dim=-1)
        div_drift =  ((-b_t*.5)*x.shape[-1]).flatten()  # TODO this is specific for VP-SDE

        assert div_drift.shape == dxt_dot_score.shape

        vel_logdet2 = div_drift + ((drift - b_t/2.* score)*score).sum(dim=1)
        vel_logdet1 = dxt_dot_score
        v_logdet = vel_logdet1+vel_logdet2
        return self.sign*(-v_logdet)
    

    def sample_and_eval_logp(self, dshape, ts, method='euler'):
        with torch.no_grad():
            device = next(self.model.parameters()).device
            x1 = torch.randn(dshape, device=device)
            plik = prior_likelihood(x1, 1.0)
            score = torch.zeros_like(x1)
            x0 = (x1, plik, score)
            solver = TorchDiffEqSolver(self)
            return solver.solve(x0, t=ts, method=method, atol=1e-5)

    def f(self, x, t):
        # Ensure t is a 1D tensor so we can broadcast
      if not torch.is_tensor(t):
          t = torch.tensor([t], dtype=x.dtype, device=x.device)
      elif t.ndim == 0:
          t = t.view(1)
      x, log_det, s = x
      x = x.to(self.dtype)      
      t_expand = t.repeat(x.size(0), 1)  # shape (batch, 1)
      score = self.model.score_func(x, t_expand) 
      return self._vel_and_vel_logdet(x, t_expand, score)


class MultiItoDensODE(ItoDensODE):
    #TODO this works currently only for probflow ODE formulation
    """Compute likelihood by Ito Density Estimator of multiple models with a single sampling model."""
   
    def __init__(self, model_sampling: DiffusionModel, models: torch.nn.ModuleList, sde: VPSDE, sign=1):
        if not isinstance(models, torch.nn.ModuleList):
            raise ValueError("models should be a torch.nn.ModuleList")
        # if not isinstance(model_sampling, DiffusionModel):
        #     raise ValueError("model_sampling should be a DiffusionModel, but is {}".format(str(model_sampling)))
        # if not isinstance(models[0], DiffusionModel):
        #     raise ValueError("model list should be list of diffusion models, but is {}".format(str(models[0])))
        super().__init__(model_sampling, sde, sign)
        self.models = models
        self.model_sampling = model_sampling
    
    def f(self, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=x.dtype, device=x.device)
        elif t.ndim == 0:
            t = t.view(1)
        x, log_dets = x
        mod_device = next(self.model_sampling.parameters()).device
        x = x.to(mod_device)
        t_expand = t.repeat(x.size(0), 1).to(mod_device)  # shape (batch, 1)
        v_sampling, v_logdet_sampling, score = self._vel_and_vel_logdet(x, t_expand, self.model_sampling.score_func(x, t_expand))
        scores = [model.score_func(x, t_expand) for model in self.models] # can be parallelized
        v_logdets = [self._vel_logdet(x, t_expand, score) for score in scores]

        self.scores =  torch.stack([score] + scores).cpu().detach()

        return v_sampling, torch.stack([v_logdet_sampling] + v_logdets)

    def sample_and_eval_logp(self, dshape, ts, method='euler'):
        with torch.no_grad():
            device = next(self.model_sampling.parameters()).device
            x1 = torch.randn(dshape, device=device)
            plik = prior_likelihood(x1, 1.0)
            plik = torch.stack([plik] + [prior_likelihood(x1, 1.0) for _ in range(len(self.models))])
            x0 = (x1, plik)
            solver = TorchDiffEqSolver(self)
            return solver.solve(x0, t=ts, method=method, atol=1e-5)


def compute_likelihood(model, x0, steps,  batched=True, 
                       mc_samples=10, device='cuda'):
    divergence_estimator = SkillingHutchinsonDivergence(mc_samples=mc_samples, batched=batched)
    pfode = LikPFODE(model.to('cuda'), model.sde, divergence_estimator=divergence_estimator)
    solver = TorchDiffEqSolver(pfode)
    dtype = next(model.parameters()).dtype

    logdet = torch.zeros((x0.size(0),), device=device)
    x0 = (x0.to(dtype), logdet.to(dtype))
    ts = torch.linspace(0, 1, steps, device='cuda').to(dtype)
    x, _ = solver.solve(x0, t=ts, method='euler', atol=1e-5)

    samples, logdet = x

    samples = samples.cpu().detach()
    logdet = logdet.cpu().detach()
    p1 = prior_likelihood(samples, 1.0)

    log_lik = p1 + logdet
    # transpose so that image makes sense
    log_lik = log_lik.T

    log_lik = log_lik.cpu().detach()
    return log_lik


def compute_likelihood_grid(model, N=150, rg=[-6,6], steps=100, device='cuda'):

    x = torch.linspace(*rg, N)
    y = torch.linspace(*rg, N)
    extent = rg+rg 

    x = torch.linspace(extent[0],extent[1], N)
    y = torch.linspace(extent[2],extent[3], N)


    X, Y = torch.meshgrid(x, y)
    x_in = torch.cat([X.reshape(-1,1), Y.reshape(-1,1)], dim=1).to(device)
    log_lik =  compute_likelihood(model, x_in, steps=steps).reshape(N,N)    
    log_lik = log_lik.T
    return log_lik




def plot_samples_with_logp(samples, logp, cmap='viridis', cutoff=-np.inf):
    """
    Plots 2D samples colored by their log probability.
    
    Parameters
    ----------
    samples : np.ndarray
        A 2D numpy array of shape (N, 2), where N is the number of samples.
    logp : np.ndarray
        A 1D array of length N, containing the log probabilities of each sample.
    cmap : str, optional
        A Matplotlib colormap name (default is 'viridis').
    """
    # Ensure samples is 2D with shape (N, 2)
    if samples.shape[1] != 2:
        raise ValueError("The function currently supports only 2D samples.")
    
    # Create a scatter plot
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        samples[:, 0][logp > cutoff],
        samples[:, 1][logp > cutoff],
        c=logp[logp > cutoff],
        cmap=cmap,
        alpha=0.8
    )
    
    # Add a colorbar to map colors to log probability
    plt.colorbar(sc, label='Log Probability')
    
    # Labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Samples Colored by Log Probability')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
