from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import logging
OmegaConf.register_new_resolver('sres', lambda x: x.replace(' ', '_'))
OmegaConf.register_new_resolver('exec', lambda x: eval(x))



@hydra.main(version_base=None, config_path="../configs", config_name="sd_fdc")
def app(cfg: DictConfig) -> None:
    # imports
    import os

    # logging level
    level = cfg.get('logging', 'INFO')
    logging.basicConfig(level=getattr(logging, level))
    # set logging level for all loggers
    logging.getLogger().setLevel(getattr(logging, level))
    logger = logging.getLogger(__name__)

    # check if slurm
    if 'SCRATCH' not in os.environ:
        os.environ['HF_HOME'] = os.path.expanduser(f'/local/home/{os.environ["USER"]}/hf')
    else:
        os.environ['HF_HOME'] = os.path.expanduser(f'/cluster/project/krause/{os.environ["USER"]}/hf')

    from maxentdiff.sd_utils import OpenClipDiversityEval

    import torch
    import os
    from maxentdiff.models import StableDiffusion
    # import code
    from maxentdiff.trainers.max_ent import MaxEntTrainer
    import torch
    from tqdm import tqdm
    from maxentdiff.sd_utils import sample_imgs, ImageSampler, OpenClipDiversityEval

    def to_rgb(image):
        return (image*255).to(torch.uint8)


    def save_ckpt(path, fine_model, base_model, optimizer, **kwargs):
        torch.save({
            'fine_model': fine_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            #'base_model': base_model.state_dict(),
            **kwargs
        },path)


    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    working_dir = cfg.get('output_dir', os.getcwd())
    ckpt_path = os.path.join(working_dir, 'last_model.ckpt')
    # print configuration with wandb
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Cuda available: {torch.cuda.is_available()}")
    logger.info(f"Cuda device count: {torch.cuda.device_count()}")
    logger.info(f"Cuda current device: {torch.cuda.current_device()}")
    logger.info(f"CUDA VISIBILE DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")




    # save config file
    OmegaConf.save(config=cfg, f=f'{working_dir}/config.yaml')

    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    cont_cfg = OmegaConf.to_container(cfg)
    cont_cfg['hydra'] = OmegaConf.to_container(hydra_config)
    wandb.init(config=cont_cfg,
               **cfg.wandb, dir=working_dir)

    
    torch.manual_seed(cfg.seed)

    fine_model: StableDiffusion = hydra.utils.instantiate(cfg.model)
    base_model: StableDiffusion =   hydra.utils.instantiate(cfg.model)
    pre_model: StableDiffusion = hydra.utils.instantiate(cfg.model)

    # encode prompt
    fine_model.encode_prompt(cfg.prompt)
    base_model.encode_prompt(cfg.prompt)
    pre_model.encode_prompt(cfg.prompt)

    trainer = hydra.utils.instantiate(cfg.trainer, model=fine_model, base_model=base_model, pre_trained_model=pre_model)
    diversity_evaluator: OpenClipDiversityEval = hydra.utils.instantiate(cfg.diversity_evaluator)
    image_sampler: ImageSampler = hydra.utils.instantiate(cfg.image_sampler, model=fine_model)

 
    print(trainer)
    #wandb.define_metric('train_step')
    #wandb.define_metric('finetune/*', step_metric='train_step')
    train_step = 0
    inner_steps = cfg.loop.training_steps // cfg.loop.md_steps
    outer_steps = cfg.loop.md_steps

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        fine_model.load_state_dict(checkpoint['fine_model'])
        base_model.load_state_dict(checkpoint['base_model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        istart, jstart = checkpoint['i'], (checkpoint['j']+1)%inner_steps
        logger.info(f"Resuming from checkpoint at step {train_step}")
    else:
        # initial evaluation
        imgs = image_sampler.get_images()
        div_met_dict = diversity_evaluator.diversity_score(list(imgs.values())[0])
        img_dict = {}
        wandb.log({'img_grid': wandb.Image(to_rgb(list(imgs.values())[0]))})
        for k,v in imgs.items():
            for i, img in enumerate(v):
                img_dict[f'{k}_{i}'] = wandb.Image(img)          
        istart, jstart = 0, 0
    # Mirror descent loop
    for mstep in range(istart,outer_steps):
        for j in tqdm(range(jstart,inner_steps), desc='Inner steps'):
            logger.info('Getting dataset...')
            dataset = trainer.get_dataset(1, verbose=False)
            logger.info('Finetuning stage...')
            finetune_loss = trainer.finetune_stage(dataset, verbose=False)
            logger.info('Finetuned!')
            if  (train_step+1) % (cfg.loop.log_freq) == 0:
                logger.info('Sampling images...')
                with torch.no_grad():
                    imgs = image_sampler.get_images()
                    logger.info('Evaluating diversity...')
                    div_met_dict = diversity_evaluator.diversity_score(list(imgs.values())[0])
                    logger.info('Diversity evaluated!')
                    logger.info('Diversity metrics: {}'.format(div_met_dict))
                    # save img grid
                wandb.log({'img_grid': wandb.Image(to_rgb(list(imgs.values())[0]))})
                for k,v in imgs.items():
                    for i, img in enumerate(v):
                        img_dict[f'{k}_{i}'] = wandb.Image(img)                        
                wandb.log(dict(**{'finetune_loss': finetune_loss}, **img_dict, **div_met_dict))
                # save_ckpt(ckpt_path, fine_model, base_model,
                #            trainer.optimizer, 
                #            train_step=train_step,
                #            i=i, j=j)
            train_step+=1
            # clear cache
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        jstart = 0
        trainer.update_base_model()
        # store model in bfloat16
        if cfg.store_model and (mstep+1) % cfg.loop.save_every == 0:
            torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_{mstep}.pt"))
        trainer.fine_model.to(torch.float32)
        # torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_{mstep}.pt"))
        # save_ckpt(ckpt_path, fine_model, base_model,
        #         trainer.optimizer, 
        #         train_step=train_step,
        #         i=mstep, j=j)
    # final evaluation
    logger.info('Sampling images...')
    with torch.no_grad():
        imgs = image_sampler.get_images()
        logger.info('Evaluating diversity...')
        div_met_dict = diversity_evaluator.diversity_score(list(imgs.values())[0])
        logger.info('Diversity evaluated!')
        wandb.log(dict(**{'finetune_loss': finetune_loss}, **img_dict, **div_met_dict))


    torch.save(trainer.fine_model.state_dict(), os.path.join(working_dir, f"model_final.pt"))


if __name__ == "__main__":
    print('Running app!')
    app()