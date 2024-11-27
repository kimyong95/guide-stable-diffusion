# I forget to log pre-train generated image during evaluation, so here's the script

from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from related_works.d3po.d3po_pytorch.diffusers_patch.pipeline_with_logprob_sdxl import pipeline_with_logprob
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from dotenv import load_dotenv
import torch
from PIL import Image
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cuda")

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
unet.load_state_dict(load_file(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")))
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", timestep_spacing="trailing")
pipeline = StableDiffusionXLPipeline.from_pretrained(model_id, unet=unet, vae=vae, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipeline.enable_vae_slicing()
pipeline = pipeline.to(device)

name_to_prompts_map = {
    "cyberdog": ["A natural fluffy dog talking to a cybertic dog."] * 16,
    "puppynose": ["Side view of a puppy lying on floor, one butterfly stopping on its nose."] * 16,
    "robotplant": ["A cute cybernetic robot plants a tree in the forest."] * 16,
    "ocean": ["A helicopter floating under the ocean."] * 16,
    "sandglass": ["A transparent glass filled with a mixture of water and sand, with one feather floating inside."] * 16,
    "penguin": ["A photo realistic photo showing a penguin standing on a very small floating ice, with a tree is on fire in the background."] * 16,
    "basket": ["Exactly one orange in a basket of apples."] * 16,
    "catbutterfly": ["A cat with butterfly wings."] * 16,
    "icecube": ["A glass of water with exactly one ice cube."] * 16,
    "trafficlight": ["A traffic light with yellow at top, green at middle, and red at bottom."] * 16,
    "deerelephant": ["A yellow reindeer and a blue elephant."] * 16,
    "apple": ["Seven red apples arranged in a circle."] * 16,
    "compress": ["elephant", "eagle", "pigeon", "hippo", "hamster", "otter", "panda", "reindeer", "owl", "penguin", "flamingo", "seal", "koala", "giraffe", "parrot", "cheetah"],
    "incompress": ["elephant", "eagle", "pigeon", "hippo", "hamster", "otter", "panda", "reindeer", "owl", "penguin", "flamingo", "seal", "koala", "giraffe", "parrot", "cheetah"],
    "aesthetic": ["elephant", "eagle", "pigeon", "hippo", "hamster", "otter", "panda", "reindeer", "owl", "penguin", "flamingo", "seal", "koala", "giraffe", "parrot", "cheetah"],
}


algos = ["ddpo", "dpok", "d3po"]
names = [
    "deerelephant", "trafficlight", "apple", "cyberdog", "puppynose", "robotplant", "ocean", "sandglass", "penguin", "basket", "icecube", "catbutterfly", \
    "compress", "incompress", "aesthetic",
]

names = ["compress", "incompress", "aesthetic"]

def run_name(algo,name):
    return f"{algo}-{name}"

for name in names: 
    eval_prompts = name_to_prompts_map[name]
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(1)
    eval_images, _, _, _ = pipeline_with_logprob(
        pipeline,
        prompt=eval_prompts,
        num_inference_steps=8,
        guidance_scale=0.0,
        eta=1.0,
        output_type="pt",
        return_dict=False,
        generator=eval_generator,
    )
    for i, image in enumerate(eval_images):
        pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        pil = pil.resize((512, 512))
        dir_ = f"{dir_path}/ppo-pretrain-images/{name}"
        os.makedirs(dir_, exist_ok=True)
        pil.save(f"{dir_}/{i}.jpeg")