# make sure you're logged in with \`huggingface-cli login\`
from collections.abc import Iterable
import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
import torch
import argparse
import os
from torchvision.utils import make_grid, save_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from genexp.models import StableDiffusion
from genexp.metrics import vendi_score
import wandb
import logging


logger = logging.getLogger(__name__)

def sample_imgs_grid(model:StableDiffusion, prompt: str, num_imgs, seed=0,
                  guidance_scale=8, steps=30,
                  device='cuda'):
    prompt = [prompt] * num_imgs
    diff_model = model.to(device)
    prompt = prompt * num_imgs
    # number of sampling steps
    steps = steps
    # guidance scale
    w = guidance_scale
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = diff_model.pipe(prompt, generator=generator, num_inference_steps=steps, guidance_scale=w, output_type='tensor')
    image = out.images
    image_grid = make_grid(torch.from_numpy(image).permute(0, 3, 1, 2), nrow=int(np.sqrt(len(image))))
    # plt.imshow(image_grid.permute(1, 2, 0))
    return image_grid

from tqdm import tqdm
def to_rgb(image):
    return (image*255)#.to(torch.uint8)
def sample_imgs(model:StableDiffusion, prompt: str, num_imgs, seed=0,
                  guidance_scale=8,
                  device='cuda', steps=30, iterations=1,
                  inception=None, clip=None, text_feat=None):
    "Samples images for a particular prompt"
    if inception:
        inception.to(device)
    if clip:
        clip.to(device)


    imgs = []
    clip_features = []
    inception_features = []
    prompt = [prompt] * num_imgs
    negative_prompt = [''] * num_imgs
    diff_model = model.to(device)
    logger.debug('running {} iterations'.format(iterations))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(iterations):
        # guidance scale
        w = guidance_scale
        logger.debug(f'guidance scale: {w}')
        logger.debug(f'sampling with seed: {seed}')
        logger.debug(f'device: {diff_model.pipe._execution_device}')
        out = diff_model.pipe(prompt=prompt, negative_prompt=negative_prompt,
                              generator=generator, num_inference_steps=steps, guidance_scale=w, 
                            output_type='pt')

        logger.debug(f'out shape: {out.images.shape}')
        images = out.images
        if clip:
            images = images.to(device)
            inception_features.append(inception.get_img_features(imgs.cuda().float()).cpu().detach())
            clip_features.append(clip.get_img_features(to_rgb(imgs.to('cuda').float())).cpu().detach())
        imgs.append(images.cpu())
    images = torch.cat(imgs, 0)
    if clip:
        return images, torch.cat(clip_features, 0), torch.cat(inception_features, 0)
    else:
        return images





from torchvision.transforms import Compose
import os
# test if on slurm cluster
if os.path.exists('/scratch/'):
    cache_dir = f'/{os.environ["SCRATCH"]}/open_clip_cache'
else:
    cache_dir = f'/local/home/{os.environ["USER"]}/open_clip_cache'
import open_clip


class ImageSampler(torch.nn.Module):
    def __init__(self, model: StableDiffusion, 
                 prompt: str,
                 device='cuda',
                 guidance_scale=8,
                 steps=30,
                 num_images=5,
                 seed=0):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.prompt = prompt
        self.num_images = num_images
        self.seed = seed
    
    def _get_images(self, n, seed=None):
        # check that n is an integer
        assert isinstance(n, int), 'Number of images should be an integer, but got {}'.format(n)
        if seed is None:
            seed = self.seed
        imgs = sample_imgs(self.model, self.prompt, n, seed, self.guidance_scale, self.device, self.steps)
        # images are (B, C, H, W)
        assert imgs.ndim == 4
        assert imgs.shape[1] == 3, 'Images should be (b,c,h,w), but got shape {}'.format(imgs.shape)
        # imgs = imgs.permute(0, 2, 3, 1)
        return imgs

    @torch.no_grad()
    def get_images(self):
        logger.debug('sampling images')
        if  isinstance(self.num_images, Iterable):
            iters, num_imgs = self.num_images
        else:
            iters = 1
            num_imgs = self.num_images
        imgs =  sample_imgs(self.model,
            prompt=self.prompt,
            num_imgs=num_imgs,
            iterations=iters,
            guidance_scale=self.guidance_scale,
            steps=self.steps
        )
        assert imgs.ndim == 4
        assert imgs.shape[1] == 3, 'Images should be (b,c,h,w), but got shape {}'.format(imgs.shape)
        return {
            self.prompt.replace(" ", "_"): imgs
        }

"""
Compose(
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    <function _convert_to_rgb at 0x79171736f060>
    ToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
)
"""
class CLIP(torch.nn.Module):
    def __init__(self, model='ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir):
        super().__init__()
        self.model, _, preprocess = open_clip.create_model_and_transforms(model, 
                                                                pretrained=pretrained, 
                                                                cache_dir=cache_dir,
                                                                )
        self.prep = Compose([preprocess.transforms[0],preprocess.transforms[1],preprocess.transforms[4]]) # extract what we need
    
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to('cuda')
        self.model.eval()
        #self.tokenizer.to('cuda')
    
    def forward(self, imgs):
        imgs = self.prep(imgs)
        return self.model.encode_image(imgs)

    @torch.no_grad()
    def get_img_features(self, imgs) -> torch.Tensor:
        imgs = self.prep(imgs)
        return self.model.encode_image(imgs)

    @torch.no_grad()
    def get_text_features(self, texts):
        text = self.tokenizer(texts)
        return self.model.encode_text(text.to('cuda'))

from pytorch_fid import inception

class InceptionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception.InceptionV3()
        self.model.eval()
    
    def forward(self, imgs):
        return self.model(imgs)

    @torch.no_grad()
    def get_img_features(self, imgs) -> torch.Tensor:
        return self.model(imgs)[0]


class CLIPInceptionDiversityEval:
    def __init__(self, model='ViT-B-32', 
                 pretrained='laion2b_s34b_b79k', 
                 cache_dir=cache_dir,
                 rbf_gamma=10.0):
        # self.clip, _, preprocess = open_clip.create_model_and_transforms(model, 
        #                                                         pretrained=pretrained, 
        #                                                         cache_dir=cache_dir,
        #                                                         )
        
        self.clip = CLIP().to('cuda')

        self.inception = InceptionV3()
        # self.prep = Compose([preprocess.transforms[0],preprocess.transforms[1],preprocess.transforms[4]]) # extract what we need
    
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip.to('cuda')
        self.inception = self.inception.to('cuda')
        self.rbf_gamma = rbf_gamma
        
    
    @torch.no_grad()
    def get_img_features(self, imgs):
        imgs = imgs.float()
        if imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2)
        return self.clip.get_img_features(imgs), self.inception.get_img_features(imgs).squeeze()

    @torch.no_grad()
    def get_text_features(self, texts):
        text = self.clip.tokenizer(texts)
        return self.clip.encode_text(text.to('cuda'))


    def diversity_score(self, imgs):
        imgs = imgs.float().to('cuda')
        if imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2)
        with torch.no_grad():
            clip_features, inc_features = self.get_img_features(imgs) # (B, d)
        
        logger.info(f'Computing diversity score with CLIP and Inception: {clip_features.shape}, {inc_features.shape}')

        res = {}
        for img_features, name in [(clip_features, 'clip'), (inc_features, 'inception')]:
            
            # normalize features
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            # compute cosine similarity
            sim = img_features @ img_features.t()
            # get the upper triangular part
            mask = torch.triu(torch.ones_like(sim), diagonal=1)
            sim = sim * mask
            avg_sim = sim.sum() / mask.sum()
            min_sim = (sim + 1-mask).min()
            max_sim = (sim - 1 + mask).max()

            # compute the vendi score
            logger.info('Computing Vendi score')
            vendi, vendi_rbf = vendi_score(img_features, gamma=self.rbf_gamma)

            res.update({
                f'{name}_avg_sim': avg_sim.item(),
                f'{name}_min_sim': min_sim.item(),
                f'{name}_max_sim': max_sim.item(),
                f'{name}_vendi_cosine': vendi,
                f'{name}_vendi_rbf_1.0': vendi_rbf,
            })

        return res
            

class OpenClipDiversityEval:

    def __init__(self, model='ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir):
        self.model, _, preprocess = open_clip.create_model_and_transforms(model, 
                                                                pretrained=pretrained, 
                                                                cache_dir=cache_dir,
                                                                )
        self.prep = Compose([preprocess.transforms[0],preprocess.transforms[1],preprocess.transforms[4]]) # extract what we need
    
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to('cuda')
        
        #self.tokenizer.to('cuda')
    
    @torch.no_grad()
    def get_img_features(self, imgs):
        imgs = self.prep(imgs)
        return self.model.encode_image(imgs.to('cuda'))

    @torch.no_grad()
    def get_text_features(self, texts):
        text = self.tokenizer(texts)
        return self.model.encode_text(text.to('cuda'))

    def diversity_score(self, imgs):
        imgs = imgs.float()
        if imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2)
        with torch.no_grad():
            img_features = self.get_img_features(imgs) # (B, d)
        # normalize features
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        # compute cosine similarity
        sim = img_features @ img_features.t()
        # get the upper triangular part
        mask = torch.triu(torch.ones_like(sim), diagonal=1)
        sim = sim * mask
        avg_sim = sim.sum() / mask.sum()
        min_sim = (sim + 1-mask).min()
        max_sim = (sim - 1 + mask).max()

        # compute the vendi score
        logger.info('Computing Vendi score')
        vendi, vendi_rbf = vendi_score(img_features)

        return {
            'avg_sim': avg_sim.item(),
            'min_sim': min_sim.item(),
            'max_sim': max_sim.item(),
            'vendi_cosine': vendi,
            'vendi_rbf_1.0': vendi_rbf,
        }
        
