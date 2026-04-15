

import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("plus", lambda x, y: x+y)
import gc

@hydra.main(version_base=None, config_path="../configs", config_name="sd_lik")
def app(cfg: DictConfig) -> None:
    import torch
    import os
    # check if slurm
    if 'SCRATCH' not in os.environ:
        os.environ['HF_HOME'] = os.path.expanduser(f'/local/home/{os.environ["USER"]}/hf')
    else:
        os.environ['HF_HOME'] = os.path.expanduser(f'/cluster/project/krause/{os.environ["USER"]}/hf')


    import os
    import os
    from maxentdiff.models import StableDiffusion

    # import code
    from tqdm import tqdm

    import numpy as np
    from maxentdiff.sd_utils import sample_imgs

    def to_rgb(image):
        return (image*255)#.to(torch.uint8)



    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    print('HF_HOME', os.environ['HF_HOME'])
    working_dir = os.getcwd()
    ckpt_path = os.path.join(working_dir, 'last_model.ckpt')
    # print configuration with wandb
    print(OmegaConf.to_yaml(cfg))
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Cuda device count: {torch.cuda.device_count()}")
    print(f"Cuda current device: {torch.cuda.current_device()}")
    print(f"CUDA VISIBILE DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # save config file
    OmegaConf.save(config=cfg, f=f'{working_dir}/config.yaml')

    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    cont_cfg = OmegaConf.to_container(cfg)
    cont_cfg['hydra'] = OmegaConf.to_container(hydra_config)
    # wandb.init(config=cont_cfg,
    #            **cfg.wandb, dir=working_dir)

    
    torch.manual_seed(cfg.seed)



    from maxentdiff.sd_utils import sample_imgs
    from maxentdiff.likelihood import compute_likelihood, compute_likelihood_grid
    import torch
    from maxentdiff.models import StableDiffusion
    from maxentdiff.likelihood import compute_likelihood

    # save images with torchvision
    import torchvision
    from tqdm import tqdm

    num_imgs = cfg.num_imgs
    num_imgs_per_seed = cfg.num_imgs_per_seed
    seeds = list(range(num_imgs // num_imgs_per_seed))
    prompt = cfg.prompt
    model_path = cfg.model_path
    model_name = cfg.model_name
    if model_path:
        model_path = os.path.join(model_path, model_name)
    store_features = cfg.store_features

    # create model_name from path
    #model_name =  model_path.split('/')[-3] + '_' + model_path.split('/')[-1]
    base_model =  hydra.utils.instantiate(cfg.model).to('cuda')
    model: StableDiffusion = hydra.utils.instantiate(cfg.model).to('cuda')
    if model_path:
        print('Loading model from:', model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print('No model path provided. Using default model.')

    # check if models are the same
    for p1, p2 in zip(model.parameters(), base_model.parameters()):
        # assert torch.all(p1 == p2)
        pass

    # path  = '/local/home/mvlastelica/sd_samples'
    prompt_subdir = os.path.join(cfg.exp_dir, model_name, prompt.replace(' ', '_'))

    # colored print subdir
    print(f"\033[1;32;40m Subdir: {prompt_subdir} \033[0m")
    print(f"\033[1;32;40m Sampling images for prompt: {prompt} \033[0m")
    
    from maxentdiff.trainers.adjoint_matching import AdjointMatchingFinetuningTrainer, sample_trajectories_ddpm

    model.encode_prompt(prompt)
    base_model.encode_prompt(prompt)

    os.makedirs(prompt_subdir, exist_ok=True)
    tqdm_iter = tqdm(seeds)
    liks_base = []
    liks_model = []
    from maxentdiff.likelihood import compute_likelihood
    for seed in  tqdm_iter:
        
        with torch.no_grad():
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            torch.manual_seed(seed)
            x1 = torch.randn(num_imgs_per_seed, 4*64*64, device=device).to(dtype)
            traj, ts = sample_trajectories_ddpm(model, x1, 30, avoid_inf=0)
            x0 = traj[:, -1]

        model = model.to('cuda')
        dtype = next(model.parameters()).dtype
        print('Computing model likelihood.')
        lik_model = compute_likelihood(model, x0.to('cuda').reshape(num_imgs_per_seed,-1).to(dtype), 
                        30, 
                        mc_samples=cfg.mc_samples, 
                        batched=True)
        liks_model.append(lik_model.cpu().detach())
        print('Computing base model likelihood.')
        lik_base_model = compute_likelihood(base_model, x0.to('cuda').reshape(num_imgs_per_seed,-1).to(dtype), 
                        30, 
                        mc_samples=cfg.mc_samples, 
                        batched=True)
        liks_base.append(lik_base_model.cpu().detach())
        print(torch.all(lik_model == lik_base_model))
        tqdm_iter.set_postfix({'generated':  (seed+1) * x0.shape[0]})
        print(f"Seed {seed} model likelihood: {lik_model.mean().item()} base model likelihood: {lik_base_model.mean().item()}")
        if  seed % 10 == 0:
            torch.save(torch.cat(liks_model, 0), os.path.join(prompt_subdir, 'liks_model.pt'))
            torch.save(torch.cat(liks_base, 0), os.path.join(prompt_subdir, 'liks_base.pt'))
            gc.collect()

    torch.save(torch.cat(liks_model, 0), os.path.join(prompt_subdir, 'liks_model.pt'))
    torch.save(torch.cat(liks_base, 0), os.path.join(prompt_subdir, 'liks_base.pt'))
    gc.collect()


if __name__ == "__main__":
    print('Running app!')
    app()