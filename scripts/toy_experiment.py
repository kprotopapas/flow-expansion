import os
# check if slurm
if 'SCRATCH' not in os.environ:
    os.environ['HF_HOME'] = os.path.expanduser(f'/local/home/{os.environ["USER"]}/hf')
else:
    os.environ['HF_HOME'] = os.path.expanduser(f'/cluster/project/krause/{os.environ["USER"]}/hf')

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from maxentdiff.solvers import TorchDiffEqSolver, PFODE, VPSDE, DDIMSolver, ODE
from maxentdiff.models import DiffusionModel
from maxentdiff.plotting import plot_density_from_points

# import code
from maxentdiff.trainers.max_ent import MaxEntTrainer
import copy
import torch
from torch import nn
from tqdm import tqdm
from maxentdiff.trainers.adjoint_matching import sample_trajectories_ddim, sample_trajectories_ddpm
from matplotlib import pyplot as plt
from maxentdiff.solvers import TorchDiffEqSolver, PFODE, VPSDE, DDIMSolver
from maxentdiff.models import DiffusionModel
from maxentdiff.likelihood import LikPFODE, SkillingHutchinsonDivergence
from maxentdiff.plotting import plot_density_from_points
from maxentdiff.likelihood import prior_likelihood
from matplotlib import pyplot as plt
from maxentdiff.utils import discrete_entropy
import numpy as np
from maxentdiff.solvers import TorchDiffEqSolver, PFODE, VPSDE, DDIMSolver
from maxentdiff.models import DiffusionModel
from maxentdiff.likelihood import LikPFODE, SkillingHutchinsonDivergence, compute_likelihood
from maxentdiff.plotting import plot_density_from_points
from maxentdiff.likelihood import prior_likelihood
from matplotlib import pyplot as plt

def likelihood_reward(model, x0):

    divergence_estimator = SkillingHutchinsonDivergence(3)
    pfode = LikPFODE(model.to('cuda'), model.sde, divergence_estimator=divergence_estimator)
    solver = TorchDiffEqSolver(pfode)
    device = 'cuda'
    logdet = torch.zeros((x0.size(0),), device=device)

    ts = torch.linspace(0, 1, 200, device='cuda')
    x, _ = solver.solve((x0, logdet), t=ts, method='rk4', atol=1e-5)
    samples, logdet = x
    samples = samples.cpu().detach()
    logdet = logdet.cpu().detach()
    p1 = prior_likelihood(samples, 1.0)
    return (p1 + logdet)


def log_likelihood_reward(trainer, original_model, step):

    device = 'cuda'
    x1 = torch.randn(3000, 2, device=device)
    trainer.fine_model.to(device)
    trainer.base_model.to(device)
    trajs, _ = sample_trajectories_ddpm(trainer.fine_model, x1, 100, sample_jumps=False)
    x0 = trajs[:, -1].to('cuda')

    ll_fine = likelihood_reward(trainer.fine_model, x0)
    ll_base = likelihood_reward(original_model, x0)
    H_fine = -(ll_fine).mean().item() # estimator for entropy
    KL = ((ll_fine-ll_base)).mean().item()
    rew_fine = - ll_fine.mean().item() 
    rew_base = - ll_base.mean().item()

    print("Rewards fine vs. base:", rew_fine, rew_base)
    wandb.log({'reward_fine': rew_fine, 'reward_base': rew_base, 'H_fine_flow': H_fine,
    'KL_fine_base': KL}, step=step)
    
def log_metrics(base_model, trainer: MaxEntTrainer, step):
    device = 'cuda'
    rg = [[-6, 6], [-6, 6]]


    with torch.no_grad():
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        x0 = torch.randn(3000, 2, device=device)
        trajs, _  = sample_trajectories_ddpm(trainer.fine_model, x0, 100)
        trajs = trajs.permute(1,0,2)
        f_counts, f_xedges, f_yedges, f_im = ax[1].hist2d(trajs[-1,:,0], trajs[-1,:,1], bins=200, range=rg)
        ax[1].set_title("Finetuned density")
        # set y lim
        x0 = torch.randn(3000, 2, device=device)

        trajs, _ = sample_trajectories_ddpm(base_model, x0, 100, sample_jumps=False)
        trajs = trajs.permute(1,0,2)
        b_counts, b_xedges, b_yedges, b_im = ax[0].hist2d(trajs[-1,:,0], trajs[-1,:,1], bins=200, range=rg)
        ax[0].set_title("Base density")

        # compute entropy
        H_base = discrete_entropy(b_counts)
        H_fine = discrete_entropy(f_counts)


        res = {'H_base': H_base, 'H_fine': H_fine, 'densities': wandb.Image(fig)}
        plt.close(fig)
        wandb.log(res, step=step)


def evaluate_reward(base_model, trainer: MaxEntTrainer, step):

    divergence_estimator = SkillingHutchinsonDivergence(20)
    pfode = LikPFODE(trainer.base_model.to('cuda'), trainer.base_model.sde, divergence_estimator=divergence_estimator)
    solver = TorchDiffEqSolver(pfode)
    device = 'cuda'

    def compute_likelihood(x0):
        ts = torch.linspace(0, 1, 100).to(device)
        logdet = torch.zeros(x0.shape[0], device=device)
        x, _ = solver.solve((x0,logdet), t=ts, method='rk4', atol=1e-5)

        samples, logdet = x
        samples = samples.cpu().detach()
        logdet = logdet.cpu().detach()
        p1 = prior_likelihood(samples, 1.0)

        log_lik = p1 + logdet
        return log_lik



    # sample from finetuned model
    from maxentdiff.trainers.adjoint_matching import sample_trajectories_ddpm
    x0 = torch.randn(3000, 2, device=device)
    trajs_fine, _  = sample_trajectories_ddpm(trainer.fine_model.to('cuda'), x0, 100)
    trajs_base, _ = sample_trajectories_ddpm(base_model.to('cuda'), x0, 100)

    rew_fine = -compute_likelihood(trajs_fine[:, -1].cuda()).mean()
    rew_base = -compute_likelihood(trajs_base[:, -1].cuda()).mean()
    print("Rewards fine vs. base:", rew_fine, rew_base)

    wandb.log({'reward_fine': rew_fine, 'reward_base': rew_base}, step=step)
        

def log_likelihood(model, name, logp=True, N=150, rg=[-6,6], step=None):

    divergence_estimator = SkillingHutchinsonDivergence(20)
    pfode = LikPFODE(model.to('cuda'), model.sde, divergence_estimator=divergence_estimator)
    solver = TorchDiffEqSolver(pfode)
    device = 'cuda'

    x = torch.linspace(*rg, N)
    y = torch.linspace(*rg, N)
    extent = rg+rg 

    x = torch.linspace(extent[0],extent[1], N)
    y = torch.linspace(extent[2],extent[3], N)


    X, Y = torch.meshgrid(x, y)
    x_in = torch.cat([X.reshape(-1,1), Y.reshape(-1,1)], dim=1).to(device)
    logdet = torch.zeros((x_in.size(0),), device=device)


    x0 = (x_in, logdet)
    ts = torch.linspace(0, 1, 200, device='cuda')
    x, _ = solver.solve(x0, t=ts, method='rk4', atol=1e-5)

    samples, logdet = x

    samples = samples.cpu().detach()
    logdet = logdet.cpu().detach()
    p1 = prior_likelihood(samples, 1.0)

    log_lik = p1.reshape(N,N) + logdet.reshape(N,N)
    # transpose so that image makes sense
    log_lik = log_lik.T

    log_lik = log_lik.cpu().detach().numpy()
    x = log_lik if logp else np.exp(log_lik)
    plt.imshow(x, cmap='hot', origin='lower', extent=extent)
    plt.gca().set_title("p_0(x_0)")
    wandb.log({name: wandb.Image(plt)}, step=step)


def save_ckpt(path, base_model, model, optimizer, i, j, **kwargs):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'base_model_state_dict': base_model.state_dict(),
        'i': i,
        'j': j,
        **kwargs
    }, path)

@hydra.main(version_base=None, config_path="../configs", config_name="toy")
def app(cfg: DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    working_dir = os.getcwd()
    ckpt_path = os.path.join(working_dir, 'ckpt')
 
    # print configuration with wandb
    print(OmegaConf.to_yaml(cfg))

    # save config file
    OmegaConf.save(config=cfg, f=f'{working_dir}/config.yaml')

    device = 'cuda'

    wandb.init(config=OmegaConf.to_container(cfg),
               **cfg.wandb, dir=working_dir, resume='auto')

    
    torch.manual_seed(cfg.seed)


    # simple mlp with sequential
    model = nn.Sequential(
        nn.Linear(3, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )

    model.to('cuda')


    # data loader for dataset
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    params = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(params)
    
    sde = VPSDE(0.1, 12, N=1000)
    diff_model = DiffusionModel(model, sde=sde)


    trainer: MaxEntTrainer = hydra.utils.instantiate(cfg.trainer, model=copy.deepcopy(diff_model), base_model=copy.deepcopy(diff_model))
    
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
            if  (train_step+1) % (cfg.loop.log_freq) == 0:
                m1 = log_metrics(diff_model, trainer, step=train_step)
                m2 = log_likelihood(trainer.fine_model, 'likelihood_fine', logp=False, step=train_step)
                m3 = log_likelihood_reward(trainer, diff_model, step=train_step)
                wandb.log({'finetune_loss': finetune_loss})
                torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_{i}_{j+1}.pt"))
                print('logging metrics!')
                save_ckpt(ckpt_path, trainer.base_model, trainer.fine_model, trainer.optimizer, i, j, train_step=train_step)
            train_step+=1
        jstart = 0
        trainer.update_base_model()
        torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_{i}.pt"))
        # print('logging metrics!')
        # log_metrics(diff_model, trainer, step=train_step)
        # m2 = log_likelihood(trainer.fine_model, 'likelihood_fine', logp=False, step=train_step)
        # m3 = log_likelihood_reward(trainer, diff_model, step=train_step)
        # torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_{i}_{j}.pt"))



    save_ckpt(ckpt_path, trainer.base_model, trainer.fine_model, trainer.optimizer, i, j)
    torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_final.pt"))



if __name__ == "__main__":
    print('Running app!')
    app()