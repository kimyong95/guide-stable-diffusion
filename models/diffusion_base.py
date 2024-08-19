
import torch
import numpy as np
import einops
import os
import math
from models.base import LightningBase
import lightning as L
from tqdm import tqdm
from lightning.pytorch.loggers.logger import DummyLogger
import torchvision
import lightning
import PIL
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from utils.finetune_difussers import FinetuneStableDiffusionPipeline, FinetuneStableDiffusion3Pipeline, FinetuneFlowMatchEulerDiscreteScheduler
from utils.finetune_difussers import FinetuneEulerDiscreteScheduler, FinetuneStableDiffusionXLPipeline
from utils.rewards import LLaVA, Gpt, Gemini, GeminiQuestion
from utils.utils import find_closest_factors, disable_train

from onediff.infer_compiler import oneflow_compile
# from DeepCache import DeepCacheSDHelper

REWAED_FUNC = {
    "llava": LLaVA,
    "gpt": Gpt,
    "gemini": Gemini,
    "gemini-question": GeminiQuestion
}

class DiffusionBase(LightningBase):

    def __init__(self, sd_model, generate_prompt, reward_query_prompt, reward_target_prompt, reward_func):

        super().__init__()
        self.sd_model = sd_model
        if sd_model == "sd2":
            self.num_sampling_steps = 50
            self.guidance_scale = 7.0
            model_id = "stabilityai/stable-diffusion-2-1-base"
            scheduler = FinetuneEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.pipe = FinetuneStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.pipe.enable_vae_slicing()
        elif sd_model == "sdxl":
            self.num_sampling_steps = 50
            self.guidance_scale = 7.0
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            scheduler = FinetuneEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.pipe = FinetuneStableDiffusionXLPipeline.from_pretrained(model_id, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.pipe.enable_vae_slicing()
            self.pipe.vae.encoder = None
        elif sd_model == "sdxl-lightning":
            self.num_sampling_steps = 8
            self.guidance_scale = 0.0
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")))
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            scheduler = FinetuneEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
            self.pipe = FinetuneStableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.pipe.enable_vae_slicing()
            self.pipe.vae.encoder = None

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

        # if not debug mode
        if not isinstance(self.trainer.logger, DummyLogger):
            self.compile()

        return self

    def compile(self):
        self.pipe.unet = oneflow_compile(self.pipe.unet)
        cache_path = f"onediff_cache/{self.sd_model}"
        try:
            self.pipe.unet.load_graph(cache_path, device=str(self.device))
        except ValueError:
            self.pipe(prompt=["hello world"], num_inference_steps=self.num_sampling_steps).images
            self.pipe.unet.save_graph(cache_path)

    def latents_to_images(self, latents):

        latents = (latents / self.pipe.vae.config.scaling_factor)
        images = self.pipe.vae.decode(latents, return_dict=False)[0]
        images = self.pipe.image_processor.postprocess(images)
        return images

    def _x_flatten(self, x):
        return einops.rearrange(x, '... C W H -> ... (C W H)', C=self.channels, W=self.sample_size, H=self.sample_size)

    def _x_unflatten(self, x):
        return einops.rearrange(x, '... (C W H) -> ... C W H', C=self.channels, W=self.sample_size, H=self.sample_size)

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
    
    def log_images(self, images, stage="train"):
        images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
        image_dir = str(self.logger.experiment.dir).removesuffix("/files") + f"/images/{stage}"
        os.makedirs(image_dir, exist_ok=True)
        grid_image = torchvision.utils.make_grid(images_tensors, nrow=find_closest_factors(len(images_tensors)))
        torchvision.utils.save_image(grid_image, f"{image_dir}/{self.current_epoch}.jpg", format='jpeg')
