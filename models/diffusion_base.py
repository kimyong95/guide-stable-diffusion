
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

from utils.finetune_difussers import FinetuneStableDiffusionPipeline, FinetuneDPMSolverMultistepScheduler
from utils.rewards import LLaVA
from utils.utils import find_closest_factors, disable_train

REWAED_FUNC = {
    "llava": LLaVA,
}

class DiffusionBase(LightningBase):

    def __init__(self, generate_prompt, reward_query_prompt, reward_target_prompt, reward_func):

        super().__init__()

        model_id = "stabilityai/stable-diffusion-2-1-base"
        self.scheduler = FinetuneDPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", algorithm_type="sde-dpmsolver++")
        self.pipe = FinetuneStableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler, torch_dtype=torch.float16)
        
        self.pipe.unet = disable_train(self.pipe.unet)
        self.pipe.text_encoder = disable_train(self.pipe.text_encoder)
        self.pipe.vae = disable_train(self.pipe.vae)

        self.sample_size = self.pipe.unet.config.sample_size
        self.channels = self.pipe.unet.config.in_channels
        
        self.generate_prompt = generate_prompt
        self.reward_query_prompt = reward_query_prompt
        self.reward_target_prompt = reward_target_prompt

        assert reward_func in REWAED_FUNC
        self.reward_func = REWAED_FUNC[reward_func]()

    def on_fit_start(self):
        super().on_fit_start()
        self.pipe = self.pipe.to(self.device)
        self.reward_func = self.reward_func.to(self.device)
        return self

    def latents_to_images(self, latents):
        images = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = self.pipe.image_processor.postprocess(images)
        return images

    # maximize reward
    # minimize total_scores
    # input: PIL images
    def get_scores(self, images):

        total_scores, outputs = self.reward_func(images, self.reward_target_prompt, self.reward_query_prompt)
        total_scores = - total_scores

        return total_scores, outputs

    def _x_flatten(self, x):
        return einops.rearrange(x, '... C W H -> ... (C W H)', C=self.channels, W=self.sample_size, H=self.sample_size)

    def _x_unflatten(self, x):
        return einops.rearrange(x, '... (C W H) -> ... C W H', C=self.channels, W=self.sample_size, H=self.sample_size)

    def log_score(self, scores, stage="train"):
        
        self.log(f"{stage}/score_mean", scores.mean())
    
    def log_images(self, images, subfix=""):
        images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
        image_dir = str(self.logger.experiment.dir).removesuffix("/files") + f"/images"
        os.makedirs(image_dir, exist_ok=True)
        grid_image = torchvision.utils.make_grid(images_tensors, nrow=find_closest_factors(len(images_tensors)))
        torchvision.utils.save_image(grid_image, f"{image_dir}/{self.current_epoch}{subfix}.jpg", format='jpeg')
