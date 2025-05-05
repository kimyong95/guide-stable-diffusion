
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
import bisect
from utils.utils import FunctionCallTracker, disable_train
from diffusers import UNet2DModel
import gpytorch
from models.diffusion_base import find_closest_factors
from palettable.colorbrewer.qualitative import Dark2_4
colors = Dark2_4.mpl_colors

from models.guidance_models import GpGuidanceModel, NnGuidanceModel

class GuideDiffusionWithOptimization(DiffusionBase):

    def __init__(self, lr_mu, lr_sigma, projection, training_batch_size, validation_batch_size, compile, number_objectives, sd_model, generate_prompt, reward_query_prompt, reward_target_prompt, reward_func, max_reward_value):
        super().__init__(sd_model, generate_prompt, reward_func, reward_query_prompt, reward_target_prompt, max_reward_value, compile)

        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.projection = projection
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.number_objectives = number_objectives

        mu = torch.zeros(self.num_sampling_steps+1, self.channels * self.sample_size * self.sample_size, dtype=torch.float32, requires_grad=False)
        self.register_buffer('mu', mu)
        sigma = torch.ones(self.num_sampling_steps+1, self.channels * self.sample_size * self.sample_size, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sigma', sigma) # entries is variance

        ###################### select objectives timestep ######################
        # self.objective_timesteps = torch.floor( (torch.arange(number_objectives)+1) / number_objectives * (self.num_sampling_steps) )
        
        self.pipe.scheduler.set_timesteps(self.num_sampling_steps)
        init_variance = torch.tensor([self.pipe.scheduler.init_noise_sigma]) ** 2
        variances = []
        stds = []
        for t in self.pipe.scheduler.timesteps:
            prev_t = t - self.pipe.scheduler.config.num_train_timesteps // self.pipe.scheduler.num_inference_steps
            variance = self.pipe.scheduler._get_variance(t, prev_t)
            eta = 1.0
            variance_t = eta * variance
            variances.append(variance_t)
            stds.append(variance_t ** 0.5)
        variances = torch.stack(variances)
        stds = torch.stack(stds)

        var_cum = torch.cat([init_variance, variances], dim=0).cumsum(dim=0)
        var_cum = var_cum / var_cum[-1]
        std_cum = torch.cat([init_variance**0.5, stds], dim=0).cumsum(dim=0)
        std_cum = std_cum / std_cum[-1]
        
        self.objective_timesteps = torch.abs(std_cum.unsqueeze(0) -  ( (torch.arange(number_objectives)+1)/number_objectives).unsqueeze(1) ).argmin(dim=1)
        self.timestep_objectives = torch.bucketize(torch.arange(self.num_sampling_steps+1), self.objective_timesteps, right=False)
        ###################### timestep-objective mapping ######################


        # dummy parameters for pytorch lightning optimizer to work
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.automatic_optimization = False

    
    def get_noise(self, batch_size, mu, sigma, generator=None):
        batch_mu = einops.repeat(mu, 'T D -> T B D', B=batch_size)

        batch_sigma = einops.repeat(sigma, 'T D -> T B D', B=batch_size)

        # to make sure noise generation same as baselines (DDPO,DPOK) for consistance comparison
        batch_noise_original = []
        for t in range(batch_mu.size()[0]):
            batch_noise_original.append(
                torch.randn(batch_mu.size()[1:], device=self.device, generator=generator)
            )
        batch_noise_original = torch.stack(batch_noise_original)
        # batch_noise_original = torch.randn(batch_mu.size(), device=self.device, generator=generator)

        batch_noise = batch_mu + batch_sigma**0.5 * batch_noise_original

        batch_noise_original_norm = batch_noise_original.norm(dim=-1)
        batch_noise_norm = batch_noise.norm(dim=-1)
        batch_noise_projected = batch_noise / batch_noise_norm[:,:,None] * batch_noise_original_norm[:,:,None]

        batch_noise = self._x_unflatten(batch_noise)
        batch_noise_projected = self._x_unflatten(batch_noise_projected)

        return batch_noise, batch_noise_projected

    def update_parameters(self, z, scores):
        ''' minimize score '''

        # z: (T, B, D)
        # scores: (T, B)

        assert z.shape[0] == scores.shape[0] == self.mu.shape[0] == self.sigma.shape[0]
        assert z.shape[1] == scores.shape[1]

        T_dim = z.shape[0]
        B_dim = z.shape[1]
        D_dim = self.mu.shape[1]

        for t in range(T_dim):
            
            scores_t = scores[t:].mean(0)
            scores_t_normalized = (scores_t - scores_t.mean()) / (scores_t.std() + 1e-8)

            w = torch.exp( - scores_t_normalized) / torch.exp( - scores_t_normalized).sum()

            self.sigma[t] = 1 / (
                
                1/self.sigma[t] + self.lr_sigma / math.sqrt(D_dim) * (
                    (1/self.sigma[t])[None,:] * (z[t] - self.mu[t,None,:]) * (z[t] - self.mu[t,None,:]) * (1/self.sigma[t])[None,:] * \
                    
                    w[:,None]

                # mean over N
                ).mean(0)
            )
            
            self.mu[t] = self.mu[t] - self.lr_mu / math.sqrt(D_dim) * (

                (z[t] - self.mu[t][None,:]) * \
                
                scores_t_normalized[:,None]
            
            # mean over N
            ).mean(0)
    
    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size
        
        epsilon, epsilon_projected = self.get_noise(batch_size, self.mu, self.sigma)

        prior = epsilon_projected[0]
        given_noise = epsilon_projected[1:]

        # collect latents trajectory
        latents_trajectory = [prior]
        def callback_func(self, index, timestep, callback_kwargs):
            latent = callback_kwargs["latents"]
            latents_trajectory.append(latent)
            return {}

        all_images = []

        final_images = self.pipe(
            [self.generate_prompt]*batch_size,
            latents=prior.type(torch.float16),
            output_type="pil",
            given_noise=given_noise,
            num_inference_steps=self.num_sampling_steps,
            guidance_scale=self.guidance_scale,
            callback_on_step_end=callback_func,
        ).images

        # deterministically sample x_{k -> K} for k in [objective_timesteps \ K]
        for i in self.objective_timesteps[:-1].int().tolist():
            
            zero_sigma = self.sigma * 1e-8
            _, shift_projected = self.get_noise(batch_size, self.mu, zero_sigma, generator=None)
            start_at_i = i
            start_at_i_latents = latents_trajectory[i]

            images_i_to_K = self.pipe(
                [self.generate_prompt]*batch_size,
                latents=prior.type(torch.float16),
                output_type="pil",
                given_noise=shift_projected[1:],
                num_inference_steps=self.num_sampling_steps,
                guidance_scale=self.guidance_scale,
                start_at_i=start_at_i,
                start_at_i_latents=start_at_i_latents,
            ).images
            all_images.append(images_i_to_K)
        
        all_images.append(final_images)

        scores = torch.zeros((self.number_objectives, batch_size), device=self.device)
        
        texts_list = []
        for i in range(self.number_objectives):
            score, text = self.get_scores(all_images[i])
            scores[i] = score
            texts_list.append(text)

            self.log(f"score_f{i}_mean", score.mean())
        
        self.log_images(all_images[-1], stage="train")
        self.log_score(scores[-1], stage="train")
        self.log_ablation(images_list=all_images, texts_list=texts_list, scores_list=scores, stage="train")

        propagated_scores = torch.zeros((self.num_sampling_steps+1, batch_size), device=self.device)
        propagated_scores = scores[self.timestep_objectives]
        self.update_parameters(self._x_flatten(epsilon), propagated_scores)
        self.log_params(self.mu, self.sigma)
        
        # dummy
        self.optimizers().step()

    def log_ablation(self, images_list=None, texts_list=None, scores_list=None, stage="train"):
        path = str(self.trainer.logger.experiment.dir).removesuffix("/files") + f"/ablation/{stage}/{self.current_epoch}"
        os.makedirs(path, exist_ok=True)

        ############################ save traj images ############################
        if images_list is not None:
            for l, images in enumerate(images_list):
                images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
                grid_image = torchvision.utils.make_grid(images_tensors, nrow=find_closest_factors(len(images_tensors)))
                torchvision.utils.save_image(grid_image, f"{path}/l={l}.jpg", format='jpeg')
        ############################ save traj images ############################

        ############################ save final text ############################
        if texts_list is not None and scores_list is not None:
            for l, (texts, scores) in enumerate(zip(texts_list, scores_list)):
                with open(f"{path}/response_l={l}.txt", "w") as f:
                    for score, text in zip(scores, texts):
                        text = text.replace('\n', '').strip()
                        f.write(f"Score: {score.item():.2f}, Text: {text}\n")
        ############################ save final text ############################


    def log_params(self, mu, sigma):
        for i, sigma_ in enumerate(sigma):
            self.log(f"||sigma_{i}_trace||_2", sigma_.sum())
        self.log(f"||mu||_2", mu.norm())

    def test_step(self, _, __):
        return

    @torch.no_grad()
    def validation_step(self, _, __):
    
        batch_size = self.validation_batch_size

        generator = torch.Generator(device=self.device).manual_seed(1)
        
        epsilon, epsilon_projected = self.get_noise(batch_size, self.mu, self.sigma, generator=generator)

        prior = epsilon_projected[0]
        given_noise = epsilon_projected[1:]

        images = self.pipe(
            [self.generate_prompt]*batch_size,
            latents=prior.type(torch.float16),
            output_type="pil",
            given_noise=given_noise,
            num_inference_steps=self.num_sampling_steps,
            guidance_scale=self.guidance_scale,
        ).images
        
        self.log_images(images, stage="validation")

        scores, texts = self.get_scores(images)
        self.log_score(scores, stage="validation")

        self.log_ablation(images_list=None, texts_list=[texts], scores_list=[scores], stage="validation")

    def configure_optimizers(self):
        return torch.optim.Adam([self.dummy], lr=0.)
