
import torch
import numpy as np
import einops
import os
import math
from models.base import LightningBase
import lightning as L
from tqdm import tqdm
import torchvision
import lightning
import PIL
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
import open_clip
from PIL import Image

from utils.finetune_difussers import FinetuneStableDiffusionPipeline, FinetuneDPMSolverMultistepScheduler

def find_closest_factors(n):
    # Start from the square root of n and move downwards to find the closest factors
    for i in range(int(math.sqrt(n)), n):
        if n % i == 0:
            return i

class ClipSimilarity:

    def __init__(self, prompt):

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()
        self.model.train = LightningBase.disabled_train_func
        self.model.requires_grad_(False)
        
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_features = self.model.encode_text(tokenizer([prompt]))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
    
    # input: PIL images
    def __call__(self, images):
        proprocessed_images = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        image_features = self.model.encode_image(proprocessed_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        cosine_similarity = einops.einsum(image_features, self.text_features, 'B D, B D -> B')
        return cosine_similarity
    
    def to(self, device):
        self.model.to(device)
        self.text_features = self.text_features.to(device)
        self.device = device
        return self

class DiffusionBase(LightningBase):

    def __init__(self, prompt,target_path):

        super().__init__()

        model_id = "stabilityai/stable-diffusion-2-1-base"
        self.scheduler = FinetuneDPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", algorithm_type="sde-dpmsolver++")
        self.pipe = FinetuneStableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler, torch_dtype=torch.float16)
        
        self.sample_size = self.pipe.unet.config.sample_size
        self.channels = self.pipe.unet.config.in_channels
        self.target = torchvision.transforms.ToTensor()(PIL.Image.open(target_path).convert('RGB'))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0,1.0), reduction=None)

        self.prompt = prompt
        self.clip_similarity = ClipSimilarity(prompt)

    def on_fit_start(self):
        super().on_fit_start()
        self.target = self.target.to(self.device)
        self.pipe = self.pipe.to(self.device)
        self.clip_similarity = self.clip_similarity.to(self.device)
        return self

    # maximize cosine similarity, maximize clip similarity
    # minimize total_scores
    def get_scores(self, images):        
        # l2_similarity = self.l2_similarity(images)
        # clip_similarity = self.clip_similarity(images)

        # scale the l2_similarity to be in the same range as clip_similarity
        # total_scores = - (l2_similarity + clip_similarity)

        total_scores = - self.clip_similarity(images)
        return total_scores

    # output range [-1, 1]
    def l2_similarity(self, images):
        images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images]).to(self.device)
        target = self.target[None,:]
        images_tensors_flatten = einops.rearrange(images_tensors, 'B C W H -> B (C W H)')
        target_flatten = einops.rearrange(target, 'B C W H -> B (C W H)')
        # Calculate the maximum possible L2 distance (for perfect dissimilarity)'
        max_distance = np.sqrt(np.prod(target_flatten.shape[1:]))
        dist = (target_flatten - images_tensors_flatten).norm(2, dim=[1])
        l2_sim = 1 - 2*(dist / max_distance)

        return l2_sim

    # output range [-1, 1]
    def cosine_similarity(self, images):
        images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images]).to(self.device)
        self.target[None,:]

        images_tensors_flatten = einops.rearrange(images_tensors, 'B C W H -> B (C W H)')
        images_tensors_flatten /= images_tensors_flatten.norm(dim=-1, keepdim=True)
        target_flatten = einops.rearrange(self.target[None,:], 'B C W H -> B (C W H)')
        target_flatten /= target_flatten.norm(dim=-1, keepdim=True)

        consine_sim = einops.einsum(images_tensors_flatten, target_flatten, 'B D, B D -> B')

        return consine_sim

    def log_score(self, scores, stage="train"):
        
        self.log(f"{stage}/score_mean", scores.mean())
    
    def log_images(self, images):
        images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            image_dir = self.logger.experiment.dir.rstrip("/files") + f"/images"
        else:
            image_dir = "debug_images"
        os.makedirs(image_dir, exist_ok=True)
        grid_image = torchvision.utils.make_grid(images_tensors, nrow=find_closest_factors(len(images_tensors)))
        torchvision.utils.save_image(grid_image, f"{image_dir}/{self.global_step}.jpg", format='jpeg')
