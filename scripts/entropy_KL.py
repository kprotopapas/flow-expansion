import os
# check if slurm
if 'SLURM_JOB_ID' not in os.environ:
    os.environ['HF_HOME'] = os.path.expanduser(f'/local/home/{os.environ["USER"]}/hf')
else:
    os.environ['HF_HOME'] = os.path.expanduser(f'/cluster/project/krause/{os.environ["USER"]}/hf')

os.environ['HF_HOME'] = os.path.expanduser(f'/local/home/{os.environ["USER"]}/hf')

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
from maxentdiff.utils import discrete_entropy
from maxentdiff.trainers.max_ent_KL_reg import MaxEntKLRegTrainer
from tqdm import tqdm

# AUXILIARY METHODS


def estimate_entropy_from_model(
    model,
    n_samples = 10000,
    traj_len = 1000,
    device ="cuda",
):
    model = model.to(device).eval()
    x0 = torch.randn(n_samples, 2, device=device)
    log_liks = compute_likelihood(model, x0, steps=traj_len)
    entropy_estimate = - log_liks.mean()
    
    return entropy_estimate.item()


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

def log_metrics(base_model, trainer: MaxEntKLRegTrainer, step):
    """TODO Use this example to handle plotting."""
    
    device = 'cuda'
    rg = [[-6, 6], [-6, 6]]


    with torch.no_grad():
        fig, ax = plt.subplots(1,2, figsize=(5,2.5), dpi=100)
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
        wandb.log(res, step=step, commit=True)


def log_everything(diff_model: DiffusionModel, trainer: MaxEntKLRegTrainer, step):
    """Log everything to wandb. TODO : make this more general / adjust for other trainers."""
    _ = log_metrics(diff_model, trainer, step=step)
    print('Logged metrics!')



# MAIN METHOD - specify config file
@hydra.main(version_base=None, config_path="../configs", config_name="entropy_KL.yaml")
def app(cfg: DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    working_dir = cfg.get('output_dir', os.getcwd())
    os.makedirs(working_dir, exist_ok=True)
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

    trainer: MaxEntKLRegTrainer = hydra.utils.instantiate(cfg.trainer, model=diff_model, base_model=diff_model_base, pre_trained_model=diff_model_pre)

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
