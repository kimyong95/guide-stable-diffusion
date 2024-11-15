
import torch
import numpy as np
import einops
import os
import math
from models.base import LightningBase
from tqdm import tqdm
import torchvision
import lightning
import PIL
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
from models.diffusion_base import DiffusionBase
from botorch.fit import fit_gpytorch_mll

from utils.utils import FunctionCallTracker, disable_train
from diffusers import UNet2DModel
import gpytorch
from models.diffusion_base import find_closest_factors
from palettable.colorbrewer.qualitative import Dark2_4
colors = Dark2_4.mpl_colors

from models.guidance_models import GpGuidanceModel, NnGuidanceModel

class FinetuneDiffusionWithModel(DiffusionBase):

    def __init__(self, guidance_model, sd_model, generate_prompt, training_batch_size, validation_batch_size, alpha, reward_func, reg_mode, reg, validation_generate_prompt=None, max_reward_value=None, reward_query_prompt=None, reward_target_prompt=None, compile=False, validation_load_epsilon=None):
        super().__init__(sd_model, generate_prompt, reward_func, reward_query_prompt, reward_target_prompt, max_reward_value, compile)

        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.validation_load_epsilon = validation_load_epsilon
        self.validation_generate_prompt = validation_generate_prompt

        self.alpha = alpha
        self.reg_mode = reg_mode
        self.reg = reg

        data_x = torch.empty(0, self.channels * self.sample_size * self.sample_size)
        data_y = torch.empty(0)

        if guidance_model.startswith("gp"):
            kernel = guidance_model.split("-")[1] if len(guidance_model.split("-")) == 2 else "rbf"
            self.guidance_model = GpGuidanceModel(self.channels * self.sample_size * self.sample_size, kernel=kernel)
        elif guidance_model == "nn":
            self.guidance_model = NnGuidanceModel(input_channels = self.channels, input_size = self.sample_size, batch_size=self.training_batch_size)
        else:
            raise NotImplementedError()

        self.guidance_model._x_flatten = self._x_flatten
        self.guidance_model._x_unflatten = self._x_unflatten

        self.register_buffer("data_x", data_x)
        self.register_buffer("data_y", data_y)

        # dummy parameters for pytorch lightning optimizer to work
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.automatic_optimization = False

    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size
        
        epsilon = torch.randn([self.num_sampling_steps+1, batch_size, self.channels, self.sample_size, self.sample_size], device=self.device, dtype=torch.float32)
        epsilon_init = epsilon.clone()

        L = 1 + self.current_epoch
        images_list, latents, epsilon, pred_y_list = self.finetune_and_generate(epsilon, L, batch_size)
        self.log_params((epsilon-epsilon_init).mean(dim=[0,1]))

        self.log_images(images_list[-1], stage="train")
        scores, texts = self.get_scores(images_list[-1])

        self.add_data(self._x_flatten(latents).type(torch.float32), scores)
        self.guidance_model.update_model_data(self.data_x, self.data_y)

        self.log_ablation(images_list=images_list, texts=texts, scores=scores, y_pred_list=pred_y_list, stage="train")
        self.log_score(scores, stage="train")

        # dummy
        self.optimizers().step()

    @torch.no_grad()
    def finetune_and_generate(self, epsilon, L, batch_size, prompts=None):
        
        epsilon = epsilon.clone()
        epsilon_init = epsilon.clone()
        epsilon_init_norm = self._x_flatten(epsilon_init).norm(dim=-1)[:,:,None,None,None]

        # ablation
        images_list = []
        pred_y_list = []

        for l in range(L):

            prior = epsilon[0]
            given_noise = epsilon[1:]

            if prompts is not None:
                assert len(prompts) == batch_size
                _prompts = prompts
            else:
                _prompts = [self.generate_prompt for _ in range(batch_size)]
            
            latents = self.pipe(
                _prompts,
                latents=prior.type(torch.float16),
                output_type="latent",
                given_noise=given_noise,
                num_inference_steps=self.num_sampling_steps,
                guidance_scale=self.guidance_scale,
                callback_steps=1,
            ).images

            pred_y, derivative_y = self.guidance_model.derivative_y_wrt_x(latents.type(torch.float32))

            epsilon -= self.alpha * derivative_y

            if self.reg_mode == "projection":
                epsilon_norm = self._x_flatten(epsilon).norm(dim=-1)[:,:,None,None,None]
                epsilon = epsilon / epsilon_norm * epsilon_init_norm
            elif self.reg_mode == "delta":
                delta = epsilon - epsilon_init
                epsilon = epsilon - self.reg * delta
            elif self.reg_mode == "delta-projection":
                delta = epsilon - epsilon_init
                epsilon = epsilon - self.reg * delta
                epsilon_norm = self._x_flatten(epsilon).norm(dim=-1)[:,:,None,None,None]
                epsilon = epsilon / epsilon_norm * epsilon_init_norm
            elif self.reg_mode == "pdf":
                pdf_value = torch.exp(-0.5 * (epsilon**2).sum(dim=[2,3,4]))
                epsilon = epsilon - self.reg * pdf_value[:,:,None,None,None] * epsilon
            elif self.reg_mode == "kl":
                epsilon = epsilon - self.reg * epsilon
            elif self.reg_mode == "none":
                pass
            else:
                raise NotImplementedError()

            pred_y_list.append(pred_y.squeeze())
            images = self.latents_to_images(latents)
            images_list.append(images)
        
        pred_y_list = torch.stack(pred_y_list)
        
        return images_list, latents, epsilon, pred_y_list
        
    def log_ablation(self, images_list=None, y_pred_list=None, texts=None, scores=None, stage="train"):
        path = str(self.trainer.logger.experiment.dir).removesuffix("/files") + f"/ablation/{stage}/{self.current_epoch}"
        os.makedirs(path, exist_ok=True)

        ########################## plot traj metric ##########################
        if y_pred_list is not None:

            y_pred_list = einops.rearrange(y_pred_list, 'L B -> B L').cpu().numpy()
            
            col = find_closest_factors(len(y_pred_list))
            row = len(y_pred_list) // col
            fig, axs = plt.subplots(row, col, figsize=(col*10, row*5))

            for i, ax in enumerate(axs.flat):

                ax.set_ylabel('y_pred', color=colors[0])
                ax.plot(y_pred_list[i], color=colors[0])

            fig.tight_layout()
            fig.savefig(f"{path}/traj_metric.png")

            # save tensors dict
            torch.save({
                "y_pred": y_pred_list,
            }, f"{path}/tensors.pt")

        ########################## plot traj metric ##########################

        ############################ save traj images ############################
        if images_list is not None:
            for l, images in enumerate(images_list):
                images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
                grid_image = torchvision.utils.make_grid(images_tensors, nrow=find_closest_factors(len(images_tensors)))
                torchvision.utils.save_image(grid_image, f"{path}/l={l}.jpg", format='jpeg')
        ############################ save traj images ############################

        ############################ save final text ############################
        if texts is not None and scores is not None:
            with open(f"{path}/response.txt", "w") as f:
                for score, text in zip(scores, texts):
                    text = text.replace('\n', '').strip()
                    f.write(f"Score: {score.item():.2f}, Text: {text}\n")
        ############################ save final text ############################

    def log_params(self, mu):
        self.log(f"||mu||_2", mu.norm())

    def test_step(self, _, __):
        return

    @torch.no_grad()
    def validation_step(self, _, __):

        batch_size = self.validation_batch_size

        generator = torch.Generator(device=self.device).manual_seed(1)

        # to reproduce exactly same images as DDPO and DPOK, for paper publication purpose
        if self.validation_load_epsilon is not None:
            epsilon = torch.load(self.validation_load_epsilon, map_location=self.device)
        else:
            epsilon = torch.randn([self.num_sampling_steps+1, batch_size, self.channels, self.sample_size, self.sample_size], device=self.device, dtype=torch.float32, generator=generator)
        
        L = 1 + self.current_epoch
        images_list, latents, epsilon, pred_y_list = self.finetune_and_generate(epsilon, L, batch_size, prompts=self.validation_generate_prompt)
        self.log_images(images_list[-1], stage="validation")

        scores, texts = self.get_scores(images_list[-1])        
        self.log_ablation(images_list=images_list, texts=texts, scores=scores, y_pred_list=pred_y_list, stage="validation")
        self.log_score(scores, stage="validation")

    def configure_optimizers(self):
        return torch.optim.Adam([self.dummy], lr=0.)

    def add_data(self, x, y):
        self.data_x = torch.cat([self.data_x, x])
        self.data_y = torch.cat([self.data_y, y])