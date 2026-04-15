

import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("plus", lambda x, y: x+y)
OmegaConf.register_new_resolver('sres', lambda x: x.replace(' ', '_'))
import gc
import wandb



@hydra.main(version_base=None, config_path="../configs", config_name="sd_gen")
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
    os.makedirs(os.getcwd(), exist_ok=True)
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
    wandb.init(config=cont_cfg,
               **cfg.wandb, dir=working_dir)

    
    torch.manual_seed(cfg.seed)


    model: StableDiffusion = hydra.utils.instantiate(cfg.model)

    from maxentdiff.sd_utils import sample_imgs
    from maxentdiff.likelihood import compute_likelihood, compute_likelihood_grid
    import torch
    from maxentdiff.models import StableDiffusion

    model = StableDiffusion(guidance_scale=8).to('cuda')
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

    if cfg.store_features:
        print('Storing features!')
        from maxentdiff.sd_utils import CLIP, InceptionV3
        inception = InceptionV3().to('cuda')
        clip = CLIP().to('cuda')


    # create model_name from path
    #model_name =  model_path.split('/')[-3] + '_' + model_path.split('/')[-1]
    if model_path and model_name != 'base_model':
        print('Loading model from:', model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print('No model path provided. Using default model.')
    
    # path  = '/local/home/mvlastelica/sd_samples'
    prompt_subdir = os.path.join(cfg.exp_dir, model_name, prompt.replace(' ', '_'))

    # colored print subdir
    print(f"\033[1;32;40m Subdir: {prompt_subdir} \033[0m")
    print(f"\033[1;32;40m Sampling images for prompt: {prompt} \033[0m")
    
    img_sampler: sample_imgs = hydra.utils.instantiate(cfg.image_sampler, model=model)
    text_feat = clip.get_text_features(prompt).cpu().detach()
    text_feat = text_feat / torch.norm(text_feat, dim=-1, keepdim=True)


    os.makedirs(prompt_subdir, exist_ok=True)
    if store_features:
        clip_features = []
        inception_features = []
    tqdm_iter = tqdm(seeds)
    for seed in  tqdm_iter:
        with torch.no_grad():
            #print('Sampling images for seed:', seed)
            imgs = img_sampler(seed=seed)
            imgs = imgs.cpu()
            #print('Sampled images:', imgs.shape)
            # gc.collect()
            # torch.cuda.empty_cache()
            if not store_features:
                for i, img in enumerate(imgs):
                    torchvision.utils.save_image(img, os.path.join(prompt_subdir, f'{seed}_{i}.png'))
            else:
                inception_features.append(inception.get_img_features(imgs.cuda().float()).cpu().detach())
                clip_features.append(clip.get_img_features(to_rgb(imgs.to('cuda').float())).cpu().detach())
                print('Clip score:', torch.einsum('ij,ij->i', clip_features[-1]/torch.norm(clip_features[-1], dim=-1, keepdim=True), text_feat).mean())
                if seed % 5 == 0:
                    torch.save(torch.cat(inception_features, 0), os.path.join(prompt_subdir, 'inception_features.pt'))
                    torch.save(torch.cat(clip_features, 0), os.path.join(prompt_subdir, 'clip_features.pt'))
            tqdm_iter.set_postfix({'generated':  (seed+1) * imgs.shape[0]})
            del imgs

    inception_features = torch.cat(inception_features, 0).squeeze()
    clip_features = torch.cat(clip_features, 0)
    print('Inception features shape:', inception_features.shape)
    print('Clip features shape:', clip_features.shape)
    torch.save(inception_features, os.path.join(prompt_subdir, 'inception_features.pt'))
    torch.save(clip_features, os.path.join(prompt_subdir, 'clip_features.pt'))
    from maxentdiff.metrics import vendi_score

    # compute vendi score
    clip_cd, clip_rbf_vendi = vendi_score(clip_features)
    inception_cd, inception_rbf_vendi = vendi_score(inception_features)

    # print to 3 decimal places
    clip_cd = round(clip_cd, 3)
    clip_rbf_vendi = round(clip_rbf_vendi, 3)
    inception_cd = round(inception_cd, 3)
    inception_rbf_vendi = round(inception_rbf_vendi, 3)

    wandb.log({'clip_cd': clip_cd,
               'clip_rbf_vendi': clip_rbf_vendi,
               'inception_cd': inception_cd,
               'inception_rbf_vendi': inception_rbf_vendi})


    s = f"===== Model name: {model_name}====\n" + \
    f'Clip Vendi score: {clip_cd} {clip_rbf_vendi}\n' + \
    f'Inception Vendi score: {inception_cd} {inception_rbf_vendi}\n' + \
    "=======================\n"
    # print above in one string
    print(s)

if __name__ == "__main__":
    print('Running app!')
    app()