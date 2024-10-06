from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
import torch
import torch
import time
import torchvision
from utils.utils import disable_train, find_closest_factors
import os

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from utils.finetune_difussers import FinetuneEulerDiscreteScheduler, FinetuneStableDiffusionXLPipeline

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from onediff.infer_compiler import oneflow_compile
import argparse


parser = argparse.ArgumentParser()


# prompt = "A transparent glass vase filled with a mixture of water and sand, with a single floating feather inside."
prompt = "A white puppy sitting on a glass field."
optimal_prompt = "A white puppy with golden wings sitting on a glass field"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

parser.add_argument('--mode', type=int, default=1)
parser.add_argument('--alpha', type=float, default=2e-3, help="step size")
parser.add_argument('--target-noise-std', type=float, default=0.0, help="add noise with standard deviation to target")

args = parser.parse_args()

mode = args.mode
alpha = args.alpha
target_noise_std = args.target_noise_std


noisy_target_str = f"-noisy-target-std={int(target_noise_std)}" if target_noise_std > 0 else ""

save_dir = f"test-shift-final/{mode}{noisy_target_str}"
os.makedirs(save_dir, exist_ok=True)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")))
scheduler = FinetuneEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
pipe = FinetuneStableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
pipe.enable_vae_slicing()

pipe.unet = disable_train(pipe.unet)
pipe.text_encoder = disable_train(pipe.text_encoder)
pipe.vae = disable_train(pipe.vae)

def latents_to_images(latents):
    latents = (latents / pipe.vae.config.scaling_factor)
    images = pipe.vae.decode(latents, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images)
    return images

# pipe.unet = oneflow_compile(pipe.unet)
# cache_path = f"onediff_cache/sdxl-lightning"
# pipe.unet.load_graph(cache_path, device="cuda")

generator = torch.Generator(device="cuda").manual_seed(0)

batch_size = 1
sample_size = pipe.unet.config.sample_size
channels = pipe.unet.config.in_channels
num_sampling_steps = 8

epsilon = torch.randn([num_sampling_steps+1, batch_size, channels, sample_size, sample_size], device="cuda", dtype=torch.float32, generator=generator)
epsilon_init = epsilon.clone()
epsilon_init_norm = epsilon_init.flatten(2, 4).norm(dim=-1)[:,:,None,None,None]

latents_traj = torch.zeros_like(epsilon)
def collect_latents_traj(i,t,_latents):
    latents_traj[i+1] = _latents

latents_traj_init = latents_traj.clone()

target_latents = pipe(
    prompt=[optimal_prompt]*batch_size,
    generator=generator,
    output_type="latent",
    num_inference_steps=num_sampling_steps, guidance_scale=0,
    latents=epsilon[0].type(torch.float16),
    given_noise=epsilon[1:],
).images

if target_noise_std > 0:
    target_latents += torch.randn_like(target_latents) * target_noise_std

target_images = latents_to_images(target_latents)
target_images[0].save(f"{save_dir}/target_images.jpeg")


L = 100
save_interval = 10
mu = []
images_list = []
for l in range(L+1):

    prior = epsilon[0]
    given_noise = epsilon[1:]

    latents_traj[0] = prior
    images = pipe(
        prompt=[prompt]*batch_size,
        generator=generator,
        num_inference_steps=num_sampling_steps, guidance_scale=0,
        latents=prior.type(torch.float16),
        given_noise=given_noise,
        callback=collect_latents_traj,
        callback_steps=1,
    ).images

    if l % save_interval == 0:
        images_list.extend(images)
    
    if mode == 1:
        direction = (latents_traj[-1] - target_latents)[None,:]
    elif mode == 2:
        direction = latents_traj - target_latents[None,:]
    
    epsilon -= alpha * direction
    
    epsilon_norm = epsilon.flatten(2, 4).norm(dim=-1)[:,:,None,None,None]
    epsilon = epsilon / epsilon_norm * epsilon_init_norm

    # shift between current epsilon and epsilon_init
    mu.append((epsilon-epsilon_init).flatten(2, 4).norm(dim=-1).mean(dim=[0,1]).item())

    

pretrain_images_tensor = torchvision.transforms.ToTensor()(images_list[0])
torchvision.utils.save_image(pretrain_images_tensor, f"{save_dir}/pretrain_images.jpeg")


images_tensor = torch.stack([torchvision.transforms.ToTensor()(image) for image in images_list[1:]])
grid_image = torchvision.utils.make_grid(images_tensor, nrow=len(images_tensor))
torchvision.utils.save_image(grid_image, f"{save_dir}/images.jpeg")



# plot
import matplotlib.pyplot as plt
plt.plot(mu)
plt.xlabel("Iteration")
plt.ylabel("Mean shift")
plt.savefig(f"{save_dir}/mu.jpeg")