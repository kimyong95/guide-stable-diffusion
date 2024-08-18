
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

class FinetuneDiffusion(DiffusionBase):

    def __init__(self, generate_prompt, reward_query_prompt, reward_target_prompt, num_sampling_steps, training_batch_size, lr, reward_func):
        super().__init__(generate_prompt, reward_query_prompt, reward_target_prompt, reward_func)

        self.num_sampling_steps = num_sampling_steps
        self.training_batch_size = training_batch_size

        self.lr = lr

        mu = torch.zeros(num_sampling_steps+1, self.channels * self.sample_size * self.sample_size, dtype=torch.float32, requires_grad=False)
        self.register_buffer('mu', mu)
        sigma = torch.ones(num_sampling_steps+1, self.channels * self.sample_size * self.sample_size, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sigma', sigma) # entries is variance

        # dummy parameters for pytorch lightning optimizer to work
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.automatic_optimization = False

    def get_noise(self, batch_size):
        batch_mu = einops.repeat(self.mu, 'T D -> T B D', B=batch_size)

        batch_sigma = einops.repeat(self.sigma, 'T D -> T B D', B=batch_size)

        batch_noise = batch_mu + batch_sigma**0.5 * torch.randn_like(batch_mu, device=self.device)

        return batch_noise

    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size

        noise = self.get_noise(batch_size)

        noise = self._x_unflatten(noise)

        init_latents = noise[0]

        images = self.pipe(
            self.generate_prompt,
            num_images_per_prompt=batch_size,
            latents=init_latents.type(torch.float16),
            given_noise=noise[1:].type(torch.float16)
        ).images

        self.log_images(images)
        scores, texts = self.get_scores(images)

        self.update_parameters(self._x_flatten(noise), scores)

        self.log_score(scores, stage="train")
        self.log_params()

        # dummy
        opt = self.optimizers()
        opt.step()


    def test_step(self, _, __):
        
        return

    @torch.no_grad()
    def validation_step(self, _, __):

        return

    def log_params(self):
        log_per_t = 10
        for t in range(self.num_sampling_steps):
            if t%log_per_t != 0:
                continue

            self.log(f"||mu_{t}||_2", self.mu[t].norm())

            sigma_t = self.sigma[t]
            max_eig = sigma_t.max()
            min_eig = sigma_t.min()

            self.log(f"sigma_{t} max eigen", max_eig)
            self.log(f"sigma_{t} min eigen", min_eig)

    def configure_optimizers(self):
        return torch.optim.Adam([self.dummy], lr=0.)
    
    def update_parameters(self, z, scores):
        ''' minimize score '''

        assert z.shape[0] == self.mu.shape[0] == self.sigma.shape[0]
        assert z.shape[1] == scores.shape[0]
        
        T_dim = z.shape[0]
        N_dim = z.shape[1]
        D_dim = self.mu.shape[1]

        min_score = scores.min()
        max_score = scores.max()
        if min_score != max_score:
            h = (scores - min_score) / (max_score - min_score)
        else:
            h = (scores - min_score)

        for t in range(T_dim):

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
            