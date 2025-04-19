import torch
import einops
import os
import torch
from rewards import GeminiQuestion
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from extended_diffusers import ExtendedStableDiffusionXLPipeline
from pseudo_target_model import PseudoTargetModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import wandb
import io
import argparse
from dotenv import load_dotenv
torch.set_float32_matmul_precision("high")
load_dotenv()

parser = argparse.ArgumentParser(description="Fast Direct Demo")
parser.add_argument("--prompt", type=str, help="Prompt for generation.", default="A yellow reindeer and a blue elephant.")
parser.add_argument("--target_prompt", type=str, help="Target prompt for objective query.", default="A realistic image of one yellow reindeer with one blue elephant.")
parser.add_argument("--train", action="store_true", help="Train Fast Direct", default=False)

is_train = parser.parse_args().train
prompt = parser.parse_args().prompt
target_prompt = parser.parse_args().target_prompt

############## pre-trained model and objective #############
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")))
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler",timestep_spacing="trailing")
pipe = ExtendedStableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.vae.enable_slicing()
pipe.vae.encoder = None
pipe = pipe.to("cuda")
gemini = GeminiQuestion()
############## pre-trained model and objective #############

###################### hyper-paramters ######################
batch_size = 32
alpha = 80
total_steps = 50
max_inner_steps = 10
num_inference_steps = 8
guidance_scale = 0.0
dimension = pipe.unet.config.in_channels * pipe.unet.config.sample_size * pipe.unet.config.sample_size
###################### hyper-paramters ######################

########################### utils ###########################
@torch.inference_mode()
def latents_to_images(latents):
    shift_factor = pipe.vae.config.shift_factor if pipe.vae.config.shift_factor else 0.0
    latents = (latents / pipe.vae.config.scaling_factor) + shift_factor
    images = pipe.vae.decode(latents, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images)
    return images

def x_flatten(x):
    return einops.rearrange(x, '... C W H -> ... (C W H)', C=pipe.unet.config.in_channels, W=pipe.unet.config.sample_size, H=pipe.unet.config.sample_size)
def x_unflatten(x):
    return einops.rearrange(x, '... (C W H) -> ... C W H', C=pipe.unet.config.in_channels, W=pipe.unet.config.sample_size, H=pipe.unet.config.sample_size)
def get_norm(epsilon):
    return x_flatten(epsilon).norm(dim=-1)[:,:,None,None,None]
def merge_images_grid(image_grid):
    # Assuming image_grid is a 2D list: [[img00, img01, ...], [img10, img11, ...], ...]
    rows = len(image_grid)
    cols = len(image_grid[0])

    # Assume all images are the same size
    img_width, img_height = image_grid[0][0].size

    # Create a new blank image with correct total size
    merged_image = Image.new('RGB', (cols * img_width, rows * img_height))

    for row_idx, row in enumerate(image_grid):
        for col_idx, img in enumerate(row):
            merged_image.paste(img, (col_idx * img_width, row_idx * img_height))

    return merged_image

generator = torch.Generator(device="cuda").manual_seed(0)
########################### utils ###########################


pseudo_target_model = PseudoTargetModel(dimension=dimension, noise_level=1e-4)
pseudo_target_model = pseudo_target_model.to("cuda")

if is_train:

    wandb_run = wandb.init(
        project="fast-direct-demo",
        config={"prompt": prompt}
    )

    for step in range(total_steps):
        ####################### initialization ######################
        epsilon = torch.randn(num_inference_steps+1, batch_size, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size, device="cuda", generator=generator)
        epsilon_init = epsilon.clone()
        epsilon_init_norm = get_norm(epsilon_init)
        ####################### initialization ######################
        
        for i in range(step+1):
            ############## generate ##############
            latents = pipe(
                [prompt]*batch_size,
                latents=epsilon[0].type(torch.float16),
                given_noise=epsilon[1:],
                output_type="latent",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=1.0,
            ).images
            ############## generate ##############

            ######## pseudo-target model #########
            pseudo_target = pseudo_target_model.estimate_pseudo_target(x_flatten(latents))
            pseudo_target = x_unflatten(pseudo_target)
            ######## pseudo-target model #########

            ############### update ###############
            # scaled_alpha = alpha * max(step+1/max_inner_steps,1)
            epsilon_hat = epsilon + alpha * (pseudo_target - latents)
            epsilon = epsilon_hat / get_norm(epsilon_hat) * epsilon_init_norm
            ############### update ###############

        images = latents_to_images(latents)
        rewards, reasons = gemini(images, target_prompt)
        objective_values = - rewards.to("cuda")
        pseudo_target_model.add_model_data(x_flatten(latents), objective_values)
        
        ################ log ##############
        log_images = []
        for index, (image, objective_value, reason) in enumerate(zip(images, objective_values, reasons)):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=75)
            buffer.seek(0)
            log_images.append(wandb.Image(Image.open(buffer), file_type="jpg", caption=f"Objective Value: [{objective_value:.2f}]; Reason: [{reason}]"))

        wandb_run.log({
            "objective_value": objective_values.mean().item(),
            "images": log_images,
            "step": step,
        })
        ################ log ##############

    data_x, data_y = pseudo_target_model.get_model_data()
    data = {"data_x": data_x, "data_y": data_y}
    torch.save(data, "data.pth")
else:
    generate_batch_size = 3
    save_per = 10
    data = torch.load("data.pth")
    data_x, data_y = data["data_x"], data["data_y"]
    pseudo_target_model.add_model_data(data_x, data_y)

    # inference
    epsilon = torch.randn(num_inference_steps+1, generate_batch_size, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size, device="cuda", generator=generator)
    epsilon_init = epsilon.clone()
    epsilon_init_norm = get_norm(epsilon_init)
    all_images = []

    for step in range(total_steps):
        latents = pipe(
            [prompt]*generate_batch_size,
            latents=epsilon[0].type(torch.float16),
            given_noise=epsilon[1:],
            output_type="latent",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=1.0,
        ).images

        pseudo_target = pseudo_target_model.estimate_pseudo_target(x_flatten(latents))
        pseudo_target = x_unflatten(pseudo_target)

        epsilon_hat = epsilon + alpha * (pseudo_target - latents)
        epsilon = epsilon_hat / get_norm(epsilon_hat) * epsilon_init_norm

        images = latents_to_images(latents)

        if step % save_per == 0:
            all_images.append(images)
    
    merged_image = merge_images_grid(all_images)
    merged_image.save("output.jpg")