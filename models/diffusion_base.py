
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
import bisect
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from utils.finetune_difussers import FinetuneStableDiffusionPipeline, FinetuneStableDiffusion3Pipeline, FinetuneFlowMatchEulerDiscreteScheduler
from utils.finetune_difussers import FinetuneEulerDiscreteScheduler, FinetuneStableDiffusionXLPipeline
from utils.rewards import GeminiQuestion
from utils.utils import find_closest_factors, disable_train

REWAED_FUNC = {
    "gemini-question": GeminiQuestion
}

class DiffusionBase(LightningBase):

    def __init__(self, sd_model, generate_prompt, reward_query_prompt, reward_target_prompt, reward_func, compile):

        super().__init__()
        self.sd_model = sd_model
        if sd_model == "sd2":
            self.num_sampling_steps = 50
            self.guidance_scale = 7.0
            model_id = "stabilityai/stable-diffusion-2-1-base"
            scheduler = FinetuneEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.pipe = FinetuneStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.pipe.enable_vae_slicing()
            self.pipe_model_config = self.pipe.unet.config
        elif sd_model == "sd2-turbo":
            self.num_sampling_steps = 4
            self.guidance_scale = 0.0
            model_id = "stabilityai/sd-turbo"
            scheduler = FinetuneEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.pipe = FinetuneStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.pipe.enable_vae_slicing()
            self.pipe_model_config = self.pipe.unet.config
        elif sd_model == "sdxl":
            self.num_sampling_steps = 30
            self.guidance_scale = 7.0
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            scheduler = FinetuneEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.pipe = FinetuneStableDiffusionXLPipeline.from_pretrained(model_id, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.pipe.enable_vae_slicing()
            self.pipe.vae.encoder = None
            self.pipe_model_config = self.pipe.unet.config
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
            self.pipe_model_config = self.pipe.unet.config
        elif sd_model == "sd3":
            self.num_sampling_steps = 28
            self.guidance_scale = 7.0
            model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
            scheduler = FinetuneFlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.pipe = FinetuneStableDiffusion3Pipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            self.pipe.vae.enable_slicing()
            self.pipe.vae.encoder = None
            self.pipe_model_config = self.pipe.transformer.config
                    
        for k, c in self.pipe.components.items():
            if isinstance(c, torch.nn.Module):
                self.pipe.components[k] = disable_train(c)

        self.sample_size = self.pipe_model_config.sample_size
        self.channels = self.pipe_model_config.in_channels
        
        self.generate_prompt = generate_prompt
        self.reward_query_prompt = reward_query_prompt

        if type(reward_target_prompt) == str:
            self.reward_target_prompt = { self.num_sampling_steps: reward_target_prompt }
        elif type(reward_target_prompt) == dict:
            self.reward_target_prompt = { int(k * (self.num_sampling_steps)): v for k, v in reward_target_prompt.items()}
        elif type(reward_target_prompt) == list:
            self.pipe.scheduler.set_timesteps(self.num_sampling_steps)
            init_sigma = self.pipe.scheduler.init_noise_sigma.unsqueeze(dim=0) if hasattr(self.pipe.scheduler,"init_noise_sigma") else torch.tensor([1.0])
            sigma_cum = torch.cat([init_sigma, self.pipe.scheduler.sigmas[:-1]], dim=0).cumsum(dim=0)
            sigma_cum = sigma_cum / sigma_cum[-1]

            reward_k = []
            delta = 1 / len(reward_target_prompt)
            for i in range(1, len(reward_target_prompt)):
                k = (sigma_cum >= i * delta).nonzero()[0].item()
                reward_k.append(k)
            reward_k.append(self.num_sampling_steps)
            self.reward_target_prompt = { k: v for k, v in zip(reward_k, reward_target_prompt) }

        assert reward_func in REWAED_FUNC
        self.reward_func = REWAED_FUNC[reward_func]()

        self.compile = compile

    def on_fit_start(self):
        super().on_fit_start()
        self.pipe = self.pipe.to(self.device)
        self.reward_func = self.reward_func.to(self.device)

        if self.compile:
            self.compile_model()

        return self

    def compile_model(self):
        if self.sd_model == "sd3":
            from onediffx import compile_pipe
            compile_options = {
                "mode": "max-optimize:max-autotune:low-precision:cache-all",
                "memory_format": "channels_last",
            }

            self.pipe = compile_pipe(
                self.pipe, backend="nexfort", options=compile_options, fuse_qkv_projections=True
            )
        
            # torch._inductor.config.conv_1x1_as_mm = True
            # torch._inductor.config.coordinate_descent_tuning = True
            # torch._inductor.config.epilogue_fusion = False
            # torch._inductor.config.coordinate_descent_check_all_directions = True
            # self.pipe.transformer.to(memory_format=torch.channels_last)
            # self.pipe.vae.to(memory_format=torch.channels_last)
            # self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune", fullgraph=True)
            # self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode="max-autotune", fullgraph=True)

            # self.pipe(
            #     prompt=["hello world"], num_inference_steps=self.num_sampling_steps, guidance_scale=self.guidance_scale,
            # ).images
                        
        elif self.sd_model in ["sd2", "sd2-turbo", "sdxl","sdxl-lightning"]:
            from onediff.infer_compiler import oneflow_compile

            self.pipe.unet = oneflow_compile(self.pipe.unet)
            cache_path = f"onediff_cache/{self.sd_model}"
            try:
                self.pipe.unet.load_graph(cache_path, device=str(self.device))
            except ValueError:
                os.makedirs("onediff_cache", exist_ok=True)
                self.pipe(prompt=["hello world"], num_inference_steps=self.num_sampling_steps).images
                self.pipe.unet.save_graph(cache_path)

    def latents_to_images(self, latents):

        latents = (latents / self.pipe.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.pipe.vae.decode(latents, return_dict=False)[0]
        images = self.pipe.image_processor.postprocess(images)
        return images

    def _x_flatten(self, x):
        return einops.rearrange(x, '... C W H -> ... (C W H)', C=self.channels, W=self.sample_size, H=self.sample_size)

    def _x_unflatten(self, x):
        return einops.rearrange(x, '... (C W H) -> ... C W H', C=self.channels, W=self.sample_size, H=self.sample_size)

    # maximize reward
    # minimize scores
    # input: PIL images
    def get_scores(self, images, index=None):

        if index is None:
            index = self.num_sampling_steps
        
        reward_steps = list(self.reward_target_prompt.keys())
        if index not in reward_steps:
            index = reward_steps[bisect.bisect_right(reward_steps,index)]

        reward_target_prompt = self.reward_target_prompt[index]

        scores, outputs = self.reward_func(images, reward_target_prompt, self.reward_query_prompt)
        scores = - scores

        return scores, outputs

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
