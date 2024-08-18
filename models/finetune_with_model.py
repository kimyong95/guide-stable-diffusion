
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

    def __init__(self, guidance_model, sd_model, generate_prompt, reward_query_prompt, reward_target_prompt, training_batch_size, validation_batch_size, alpha, beta, reward_func):
        super().__init__(sd_model, generate_prompt, reward_query_prompt, reward_target_prompt, reward_func)

        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size

        self.alpha = alpha
        self.beta = beta

        data_x = torch.empty(0, self.channels * self.sample_size * self.sample_size)
        data_y = torch.empty(0)

        if guidance_model == "gp":
            self.guidance_model = GpGuidanceModel(self.channels * self.sample_size * self.sample_size)
        elif guidance_model == "nn":
            self.guidance_model = NnGuidanceModel(input_channels = self.channels, input_size = self.sample_size)
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
        images_list, latents = self.finetune_and_generate(epsilon, L, batch_size)
        self.log_params((epsilon-epsilon_init).mean(dim=[0,1]))

        self.log_images(images_list[-1])
        scores, texts = self.get_scores(images_list[-1])

        self.add_data(self._x_flatten(latents).type(torch.float32), scores)

        self.guidance_model.update_model_data(self.data_x, self.data_y)

        self.log_score(scores, stage="train")

    def finetune_and_generate(self, epsilon, L, batch_size):

        epsilon_init = epsilon.clone()
        epsilon_init_norm = self._x_flatten(epsilon_init).norm(dim=-1)[:,:,None,None,None]

        # ablation
        images_list = []

        latents_traj = torch.zeros_like(epsilon)
        for l in range(L):

            prior = epsilon[0]
            given_noise = epsilon[1:]

            latents_traj[0] = prior
            def collect_latents_traj(i,t,_latents):
                latents_traj[i+1] = _latents
            
            latents = self.pipe(
                [self.generate_prompt]*batch_size,
                latents=prior.type(torch.float16),
                output_type="latent",
                given_noise=given_noise,
                num_inference_steps=self.num_sampling_steps,
                guidance_scale=self.guidance_scale,
                callback=collect_latents_traj,
                callback_steps=1,
            ).images

            assert torch.all(latents_traj[-1] == latents)

            pred_y, derivative_y = self.guidance_model.derivative_y_wrt_x(latents.type(torch.float32))

            epsilon -= self.alpha * derivative_y + self.beta * (latents_traj - latents_traj[-1][None,:])
            epsilon_norm = self._x_flatten(epsilon).norm(dim=-1)[:,:,None,None,None]
            epsilon = epsilon / epsilon_norm * epsilon_init_norm

            images = self.latents_to_images(latents)
            images_list.append(images)
        
        return images_list, latents
        
    def log_ablation(self, images_list=None, scores_list=None, y_pred_list=None, y_uncertainty_list=None, texts=None, scores=None):
        path = str(self.trainer.logger.experiment.dir).removesuffix("/files") + f"/ablation/{self.current_epoch}"
        os.makedirs(path, exist_ok=True)

        ########################## plot traj metric ##########################
        if scores_list is not None and y_pred_list is not None and y_uncertainty_list is not None:

            traj_scores = torch.stack(scores_list)
            y_pred = torch.stack(y_pred_list)
            y_uncertainty = torch.stack(y_uncertainty_list)

            traj_scores = einops.rearrange(traj_scores, 'L B -> B L').cpu().numpy()
            y_pred = einops.rearrange(y_pred, 'L B -> B L').cpu().numpy()
            y_uncertainty = einops.rearrange(y_uncertainty, 'L B -> B L').cpu().numpy()
            
            col = find_closest_factors(len(traj_scores))
            row = len(traj_scores) // col
            fig, axs = plt.subplots(row, col, figsize=(col*10, row*5))

            for i, ax in enumerate(axs.flat):

                ax.set_ylabel('scores', color=colors[0])
                ax.plot(traj_scores[i], color=colors[0])

                ax1 = ax.twinx()
                ax1.set_ylabel('y_pred', color=colors[1])
                ax1.spines['right'].set_position(('outward', 0))
                ax1.yaxis.get_offset_text().set_position((1.1,0))
                ax1.plot(y_pred[i], color=colors[1])

                ax2 = ax.twinx()
                ax2.set_ylabel('y_uncertainty', color=colors[2])
                ax2.spines['right'].set_position(('outward', 60))
                ax2.yaxis.get_offset_text().set_position((1.3,0))
                ax2.plot(y_uncertainty[i], color=colors[2])

            fig.tight_layout()
            fig.savefig(f"{path}/traj_metric.png")

            # save tensors dict
            torch.save({
                "traj_scores": scores_list,
                "y_pred": y_pred,
                "y_uncertainty": y_uncertainty,
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
            with open(f"{path}/llava.txt", "w") as f:
                for score, text in zip(scores, texts):
                    text = text.replace('\n', '').strip()
                    f.write(f"Score: {score.item():3f}, Text: {text}\n")
        ############################ save final text ############################

    def log_params(self, mu):
        self.log(f"||mu||_2", mu.norm())

    def test_step(self, _, __):
        return

    @torch.no_grad()
    def validation_step(self, _, __):

        batch_size = self.validation_batch_size

        generator = torch.Generator(device=self.device).manual_seed(1)
        epsilon = torch.randn([self.num_sampling_steps+1, batch_size, self.channels, self.sample_size, self.sample_size], device=self.device, dtype=torch.float32, generator=generator)
        L = 1 + self.current_epoch
        images_list, latents = self.finetune_and_generate(epsilon, L, batch_size)
        scores, texts = self.get_scores(images_list[-1])

        self.log_ablation(images_list=images_list, texts=texts, scores=scores)
        self.log_score(scores, stage="validation")

    def configure_optimizers(self):
        return torch.optim.Adam([self.dummy], lr=0.)

    def add_data(self, x, y):
        self.data_x = torch.cat([self.data_x, x])
        self.data_y = torch.cat([self.data_y, y])