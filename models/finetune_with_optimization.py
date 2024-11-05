
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

class FinetuneDiffusionWithOptimization(DiffusionBase):

    def __init__(self, lr, projection, training_batch_size, validation_batch_size, compile, evaluate_intermidiate_steps, sd_model, generate_prompt, reward_query_prompt, reward_target_prompt, reward_func, max_reward_value):
        super().__init__(sd_model, generate_prompt, reward_query_prompt, reward_target_prompt, reward_func, max_reward_value, compile)

        self.lr = lr
        self.projection = projection
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.evaluate_intermidiate_steps = evaluate_intermidiate_steps

        mu = torch.zeros(self.num_sampling_steps+1, self.channels * self.sample_size * self.sample_size, dtype=torch.float32, requires_grad=False)
        self.register_buffer('mu', mu)
        sigma = torch.ones(self.num_sampling_steps+1, self.channels * self.sample_size * self.sample_size, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sigma', sigma) # entries is variance

        # dummy parameters for pytorch lightning optimizer to work
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.automatic_optimization = False


    def get_noise(self, batch_size, generator=None):
        batch_mu = einops.repeat(self.mu, 'T D -> T B D', B=batch_size)

        batch_sigma = einops.repeat(self.sigma, 'T D -> T B D', B=batch_size)

        batch_noise_original = torch.randn(batch_mu.size(), device=self.device, generator=generator)

        batch_noise = batch_mu + batch_sigma**0.5 * batch_noise_original

        batch_noise_original_norm = batch_noise_original.norm(dim=-1)
        batch_noise_norm = batch_noise.norm(dim=-1)
        batch_noise_projected = batch_noise * ( batch_noise_original_norm[:,:,None] / batch_noise_norm[:,:,None] )

        batch_noise_original = self._x_unflatten(batch_noise_original)
        batch_noise_projected = self._x_unflatten(batch_noise_projected)

        return batch_noise_projected, batch_noise_original

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
            # min max normalize score to [0,1]
            min_score = scores_t.min()
            max_score = scores_t.max()
            if min_score != max_score:
                h = (scores_t - min_score) / (max_score - min_score)
            else:
                h = (scores_t - min_score)

            _beta = self.lr
            self.sigma[t] = 1 / (
                
                1/self.sigma[t] + _beta / D_dim * (
                    (1/self.sigma[t])[None,:] * (z[t] - self.mu[t,None,:]) * (z[t] - self.mu[t,None,:]) * (1/self.sigma[t])[None,:] * \
                    
                    h[:,None]

                # mean over N
                ).mean(0)
            )
            
            self.mu[t] = self.mu[t] - _beta / math.sqrt(D_dim) * (

                (z[t] - self.mu[t][None,:]) * \
                
                h[:,None]
            
            # mean over N
            ).mean(0)
    
    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size
        
        epsilon_projected, epsilon_original = self.get_noise(batch_size)

        prior = epsilon_projected[0]
        given_noise = epsilon_projected[1:]

        # collect latents trajectory
        latents_trajectory = [prior]
        def callback_func(self, index, timestep, callback_kwargs):
            latent = callback_kwargs["latents"]
            latents_trajectory.append(latent)
            return {}

        images_final = self.pipe(
            [self.generate_prompt]*batch_size,
            latents=prior.type(torch.float16),
            output_type="pil",
            given_noise=given_noise,
            num_inference_steps=self.num_sampling_steps,
            guidance_scale=self.guidance_scale,
            callback_on_step_end=callback_func,
        ).images

        texts_trajectory = []
        images_trajectory = []

        scores = torch.zeros((self.num_sampling_steps+1, batch_size), device=self.device)

        if bool(self.evaluate_intermidiate_steps):
            if type(self.evaluate_intermidiate_steps) == bool:
                evaluation_steps = list(range(self.num_sampling_steps))
            if type(self.evaluate_intermidiate_steps) == int:
                evaluation_steps = list(range(0, self.num_sampling_steps, self.evaluate_intermidiate_steps))
        else:
            rewards_steps = list(self.reward_target_prompt.keys())
            evaluation_steps = list(set(rewards_steps) - {self.num_sampling_steps})
            evaluation_steps.sort()
        
        # xt -> xT, fot t in evaluation_steps
        for i in evaluation_steps:
            images_i = self.pipe(
                [self.generate_prompt]*batch_size,
                output_type="pil",
                start_at_i=i,
                start_at_i_latents=latents_trajectory[i] if i > 0 else None,
                latents=prior.type(torch.float16),
                num_inference_steps=self.num_sampling_steps,
                guidance_scale=self.guidance_scale,
            ).images

            scores_i, texts_i = self.get_scores(images_i, i)

            scores[i] = scores_i

            self.log(f"train/score_{i}_mean", scores_i.mean())

            images_trajectory.append(images_i)
            texts_trajectory.append(texts_i)
        
        scores_final, texts_final = self.get_scores(images_final)
        self.log_score(scores_final, stage="train")
        scores[-1] = scores_final

        evaluated_steps = evaluation_steps + [self.num_sampling_steps]

        # fill scores by future timestep
        for i in range(self.num_sampling_steps):
            if i not in evaluated_steps:
                next_i = evaluated_steps[bisect.bisect_right(evaluated_steps,i)]
                scores[i] = scores[next_i]

        images_trajectory.append(images_final)
        texts_trajectory.append(texts_final)

        self.log_images(images_trajectory[-1], stage="train")

        
        self.log_params(self.mu, self.sigma)
        self.log_ablation(images_list=images_trajectory, texts_list=texts_trajectory, scores_list=scores[evaluated_steps], stage="train")

        self.update_parameters(self._x_flatten(epsilon_original), scores)
        
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
        
        epsilon_projected, epsilon_original = self.get_noise(batch_size, generator=generator)

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
