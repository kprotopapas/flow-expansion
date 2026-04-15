import os
# check if slurm
if 'SLURM_JOB_ID' not in os.environ:
    os.environ['HF_HOME'] = os.path.expanduser(f'/local/home/{os.environ["USER"]}/hf')
else:
    os.environ['HF_HOME'] = os.path.expanduser(f'/cluster/project/krause/{os.environ["USER"]}/hf')

os.environ['HF_HOME'] = os.path.expanduser(f'/cluster/project/krause/{os.environ["USER"]}/hf')

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from maxentdiff.solvers import TorchDiffEqSolver, PFODE, VPSDE, DDIMSolver, ODE
from maxentdiff.models import DiffusionModel

from maxentdiff.plotting import plot_density_from_points
import torch
from torch import nn
from maxentdiff.solvers import TorchDiffEqSolver, PFODE, VPSDE, DDIMSolver, ODE
from maxentdiff.models import DiffusionModel
from maxentdiff.plotting import plot_density_from_points
from maxentdiff.trainers.adjoint_matching import AdjointMatchingFinetuningTrainer, sample_trajectories_ddpm
import copy
from matplotlib import pyplot as plt
import numpy as np
from maxentdiff.trainers.adjoint_matching import sample_trajectories_ddpm
from math import log, pi
from maxentdiff.likelihood import compute_likelihood
from maxentdiff.trainers.wasserstein1_functional import Wasserstein1Trainer
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.autograd as autograd

# AUXILIARY METHODS
def estimate_expectation_from_model(model, num_samples=1024):
    # estimate expectation from model
    x0 = torch.randn(num_samples, 2, device=device)
    with torch.no_grad():
        trajs, _ = sample_trajectories_ddpm(model, x0, 1000)
    final_pts = trajs[:, -1, :]  
    per_sample_losses = reward_function(final_pts)
    expectation = per_sample_losses.mean()

    return expectation

def estimate_wasserstein1_from_models(
    model_p, model_q, A_inv,
    num_samples=10240,
    traj_len=100,
    critic_steps=100,
    gp_lambda=2.0,
    critic_lr=1e-4,
    device='cuda'
):
    # 1) sample final points from each model
    with torch.no_grad():
        z_p = torch.randn(num_samples, 2, device=device)
        trajs_p, _ = sample_trajectories_ddpm(model_p, z_p, traj_len)
        x = trajs_p[:, -1, :].to(device)

        z_q = torch.randn(num_samples, 2, device=device)
        trajs_q, _ = sample_trajectories_ddpm(model_q, z_q, traj_len)
        y = trajs_q[:, -1, :].to(device)

    # 2) build a tiny critic (spectral-norm MLP)
    critic = nn.Sequential(
        nn.utils.parametrizations.spectral_norm(nn.Linear(2, 128)),
        nn.ReLU(),
        nn.utils.parametrizations.spectral_norm(nn.Linear(128, 1)),
    ).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=critic_lr, betas=(0.5,0.9))

    # 3) WGAN-GP loop
    for _ in range(critic_steps):
        opt.zero_grad(set_to_none=True)
        # dual objective (we *minimise* −gap)
        f_x = critic(x).mean()
        f_y = critic(y).mean()
        loss = -(f_x - f_y)

        # gradient penalty
        eps   = torch.rand(num_samples, 1, device=device)
        interp = eps * x + (1-eps) * y
        interp.requires_grad_(True)
        f_int = critic(interp)
        grad  = autograd.grad(
            f_int, interp,
            grad_outputs=torch.ones_like(f_int),
            create_graph=True, retain_graph=True
        )[0]

        # un-weighted gradient penalty (option A)
        # gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()  

        # weighted gradient penalty (option B)
        grad_trans = grad @ A_inv
        gp = ((grad_trans.norm(2, dim=1) - 1) ** 2).mean()

        (loss + gp_lambda*gp).backward()
        opt.step()

    # 4) final W1 estimate
    with torch.no_grad():
        w1 = critic(x).mean() - critic(y).mean()

    return w1


def reward_function(x):
    reward = torch.norm(x, dim=1)

    return reward

def grad_reward(x):
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        y = reward_function(x_).sum()
    return torch.autograd.grad(y, x_, retain_graph=True)[0]


# SAVING, PLOTTING AND LOGGING METHODS

def save_ckpt(path, base_model, model, optimizer, i, j, **kwargs):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'base_model_state_dict': base_model.state_dict(),
        'i': i,
        'j': j,
        **kwargs
    }, path)

def log_metrics(base_model, trainer: Wasserstein1Trainer, step, plot_rg=[[5,30], [5,30]]):
    """TODO Use this example to handle plotting."""
    
    device = 'cuda'
    with torch.no_grad():
        fig, ax = plt.subplots(1,2, figsize=(5,2.5), dpi=100)
        x0 = torch.randn(3000, 2, device=device)
        trajs, _  = sample_trajectories_ddpm(trainer.fine_model, x0, 100)
        trajs = trajs.permute(1,0,2)
        f_counts, f_xedges, f_yedges, f_im = ax[1].hist2d(trajs[-1,:,0], trajs[-1,:,1], bins=200, range=plot_rg)
        ax[1].set_title("Finetuned density")
        # set y lim
        x0 = torch.randn(3000, 2, device=device)

        trajs_base, _ = sample_trajectories_ddpm(trainer.pre_trained_model, x0, 100, sample_jumps=False)
        trajs_base = trajs_base.permute(1,0,2)
        b_counts, b_xedges, b_yedges, b_im = ax[0].hist2d(trajs_base[-1,:,0], trajs_base[-1,:,1], bins=200, range=plot_rg)
        ax[0].set_title("Pre-trained density")

        # compute expectation differences
        mean_x_diff = trajs[-1,:,0].mean().item() - trajs_base[-1,:,0].mean().item() 
        mean_y_diff = trajs[-1,:,1].mean().item() - trajs_base[-1,:,1].mean().item() 

        res = {'mean_x_diff': mean_x_diff, 'mean_y_diff': mean_y_diff, 'densities': wandb.Image(fig)}
        plt.close(fig)
        wandb.log(res, step=step, commit=True)


def log_everything(diff_model: DiffusionModel, trainer: Wasserstein1Trainer, step):
    """Log everything to wandb. TODO : make this more general / adjust for other trainers."""
    _ = log_metrics(diff_model, trainer, step=step)
    print('Logged metrics!')



# MAIN METHOD - specify config file
@hydra.main(version_base=None, config_path="../configs", config_name="W1.yaml")
def app(cfg: DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    working_dir = os.getcwd()
    ckpt_path = os.path.join(working_dir, 'ckpt')

    # ─── Set up W&B cache inside the real run folder ──────────────
    # cache_dir = os.path.join(working_dir, "wandb_cache")
    # os.makedirs(cache_dir, exist_ok=True)
    # os.environ["WANDB_CACHE_DIR"] = cache_dir
 
    # print configuration with wandb
    print(OmegaConf.to_yaml(cfg))

    # save config file
    OmegaConf.save(config=cfg, f=f'{working_dir}/config.yaml')

    device = 'cuda'

    wandb.init(config=OmegaConf.to_container(cfg), mode="online",
               **cfg.wandb, dir=working_dir, resume='auto')

    
    torch.manual_seed(cfg.seed)


    model1 = nn.Sequential(
        nn.Linear(3, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )

    model1.to('cuda')

    params = torch.load(cfg.model_path, map_location=device)
    model1.load_state_dict(params)
    
    N = 1000
    sde = VPSDE(0.1, 12, N)
    diff_model1 = DiffusionModel(model1, sde=sde)

    diff_model_pre = copy.deepcopy(diff_model1)
    diff_model_base = copy.deepcopy(diff_model1)
    diff_model = copy.deepcopy(diff_model1)

    K = cfg.distance_param_K
    a = torch.tensor([1.0, K], device=device)
    A_inv = torch.diag(1.0 / a) 

    trainer: Wasserstein1Trainer = hydra.utils.instantiate(cfg.trainer, model=diff_model, base_model=diff_model_base, pre_trained_model=diff_model_pre, A_inv=A_inv, grad_reward=grad_reward)


    istart = 0
    jstart = 0
    inner_steps = cfg.loop.training_steps // cfg.loop.md_steps
    outer_steps = cfg.loop.md_steps
    train_step = 0
    if os.path.exists(ckpt_path):
        print('Restarting!')
        ckpt = torch.load(ckpt_path)
        trainer.fine_model.load_state_dict(ckpt['model_state_dict'])
        trainer.base_model.load_state_dict(ckpt['base_model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        jstart = ckpt['j']+1
        istart = ckpt['i']
        train_step = ckpt['train_step']
        if jstart == inner_steps:
            jstart = 0
            istart += 1
    print(trainer)
    for i in tqdm(range(istart, outer_steps), desc='MD steps'):
        for j in tqdm(range(jstart, inner_steps)):
            dataset = trainer.get_dataset(1, verbose=False)
            finetune_loss = trainer.finetune_stage(dataset, verbose=False)
            if  train_step == 0 or (train_step+1) % (cfg.loop.log_freq) == 0:
                print('Logging metrics!')
                log_everything(diff_model=diff_model, trainer=trainer, step=train_step)
                wandb.log({'finetune_loss': finetune_loss})
                torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_{i}_{j+1}.pt"))
                print('logging metrics!')
                save_ckpt(ckpt_path, trainer.base_model, trainer.fine_model, trainer.optimizer, i, j, train_step=train_step)
            train_step+=1
        jstart = 0
        trainer.update_base_model()
    log_everything(diff_model=diff_model, trainer=trainer, step=train_step)
    save_ckpt(ckpt_path, trainer.base_model, trainer.fine_model, trainer.optimizer, i, j)
    torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_final.pt"))
    wandb.finish()


if __name__ == "__main__":
    print('Running app!')
    app()
