from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
import torch
import torch
import time
import torchvision
from utils.utils import disable_train, find_closest_factors
import os

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from utils.guide_difussers import FinetuneEulerDiscreteScheduler, FinetuneStableDiffusionXLPipeline

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from onediff.infer_compiler import oneflow_compile
import argparse

#################################
# Mode 1: Fast Direct (alpha=2e-3)
# Mode 2: Default Classifier (alpha=2e-3)
# Mode 3: MPGD (alpha=5e-5) (x_k-\mu{x_k})
# Mode 4: DPS, LGD, UGD (alpha=5e-5) d(x_k-\mu{x_k}) (derivative through neural network)
#################################

parser = argparse.ArgumentParser()


prompt = "A white puppy sitting on a glass field."
optimal_prompt = "A white puppy with golden wings sitting on a glass field"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

parser.add_argument('--mode', type=int, default=3)
parser.add_argument('--alpha', type=float, default=2e-3, help="step size")
parser.add_argument('--projection', action=argparse.BooleanOptionalAction, default=True, help="whether to project the direction")
parser.add_argument('--target-noise-std', type=float, default=0.0, help="add noise with standard deviation to target")
parser.add_argument('--mode-1-k', type=int, default=None)

args = parser.parse_args()

mode = args.mode
alpha = args.alpha
projection = args.projection
target_noise_std = args.target_noise_std
optional_str = ""

if args.mode_1_k is not None:
    optional_str = f"-k={args.mode_1_k}"


save_dir = f"test-shift-final/results/mode={mode}{optional_str}/p={projection}/std={int(target_noise_std)}"
os.makedirs(save_dir, exist_ok=True)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")))
scheduler = FinetuneEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
# scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
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
guidance_scale = 0.0

epsilon = torch.randn([num_sampling_steps+1, batch_size, channels, sample_size, sample_size], device="cuda", dtype=torch.float32, generator=generator)
epsilon_init = epsilon.clone()
epsilon_init_norm = epsilon_init.flatten(2, 4).norm(dim=-1)[:,:,None,None,None]

target_save_path = "test-shift-final/results/target_latents.pt"
if os.path.exists(target_save_path):
    target_latents = torch.load(target_save_path).to("cuda")
else:
    target_latents = pipe(
        prompt=[optimal_prompt]*batch_size,
        generator=generator,
        output_type="latent",
        num_inference_steps=num_sampling_steps,
        latents=epsilon[0].type(torch.float16),
        given_noise=epsilon[1:],
    ).images
    torch.save(target_latents, target_save_path)

if target_noise_std > 0:
    target_latents += torch.randn_like(target_latents) * target_noise_std

target_images = latents_to_images(target_latents)
target_images[0].save(f"{save_dir}/target_images.jpeg")


L = 100
save_interval = 10
mu = []
images_list = []
latents_traj = []

for l in range(L+1):

    prior = epsilon[0]
    given_noise = epsilon[1:]

    latents_traj.append(prior)
    
    def callback_func(self, index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        pred_original_sample = callback_kwargs["pred_original_sample"]

        if mode in [1,2]:
            latents_traj.append(latents)
        if mode == 3:
            latents_traj.append(pred_original_sample)
        return {
            "latents": latents
        }

    images = pipe(
        prompt=[prompt]*batch_size,
        generator=generator,
        num_inference_steps=num_sampling_steps, guidance_scale=guidance_scale,
        latents=prior.type(torch.float16),
        given_noise=given_noise,
        callback_on_step_end=callback_func,
        callback_on_step_end_tensor_inputs=["latents", "pred_original_sample"],
    ).images

    if l % save_interval == 0:
        images_list.extend(images)
        images[0].save(f"{save_dir}/images_{l}.jpeg")
    
    if mode == 1:
        if args.mode_1_k is not None:
            index_k = args.mode_1_k
        else:
            index_k = -1
        latents_traj_tensor = torch.stack(latents_traj)
        direction = (latents_traj_tensor[index_k] - target_latents)[None,:]
    elif mode == 2:
        latents_traj_tensor = torch.stack(latents_traj)
        direction = latents_traj_tensor - target_latents[None,:]
    elif mode == 3:
        # Set E[x_K | x_0] = E[x_K | x_1]
        latents_traj[0] = latents_traj[1]
        latents_traj_tensor = torch.stack(latents_traj)
        direction = latents_traj_tensor - target_latents[None,:]

    epsilon -= alpha * direction
    
    if projection:
        epsilon_norm = epsilon.flatten(2, 4).norm(dim=-1)[:,:,None,None,None]
        epsilon = epsilon / epsilon_norm * epsilon_init_norm

    # shift between current epsilon and epsilon_init
    mu.append((epsilon-epsilon_init).flatten(2, 4).norm(dim=-1).mean(dim=[0,1]).item())

    latents_traj.clear()
    

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