
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

from utils.finetune_difussers import FinetuneStableDiffusionPipeline, FinetuneDPMSolverMultistepScheduler
from utils.utils import FunctionCallTracker

from diffusers import UNet2DModel
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class FinetuneDiffusionWithModel(DiffusionBase):

    def __init__(self, prompt, num_sampling_steps, training_batch_size, lr):
        super().__init__(prompt)

        self.num_sampling_steps = num_sampling_steps
        self.training_batch_size = training_batch_size

        self.lr = lr

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = 1e-4
        self.likelihood.eval()

        data_x = torch.empty(0, self.channels * self.sample_size * self.sample_size)
        data_y = torch.empty(0)

        self.register_buffer("data_x", data_x)
        self.register_buffer("data_y", data_y)

        self.model = ExactGPModel(None, None, self.likelihood)
        self.model.covar_module.base_kernel.lengthscale = (self.channels * self.sample_size * self.sample_size) ** 0.5
        self.model.eval()
        # Get into evaluation (predictive posterior) mode

        # dummy parameters for pytorch lightning optimizer to work
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.automatic_optimization = False

    def on_fit_start(self):
        super().on_fit_start()
        self.data_x = self.data_x.to(self.device)
        self.data_y = self.data_y.to(self.device)

    @staticmethod
    def get_noise(self, x, mu):

        x = x.type(torch.float32)

        noise = mu + torch.randn_like(x)

        return noise

    @torch.enable_grad()
    def get_mu(self, x):

        x_flatten = self._x_flatten(x)
        x_flatten.requires_grad = True
        

        y_pred_grad = []
        for xj in x_flatten:
            yj = self.likelihood(self.model(xj.unsqueeze(0)))
            mean = yj.mean
            lcb = yj.mean - 2*(yj.covariance_matrix.item()**0.5)
            loss = mean + 0.5 * lcb
            grad = torch.autograd.grad(mean, xj, retain_graph=False, allow_unused=True)[0]
            grad = torch.zeros_like(xj) if grad is None else grad
            y_pred_grad.append(grad)
        
        y_pred_grad = torch.stack(y_pred_grad)

        y_pred_grad = self._x_unflatten(y_pred_grad)

        mu = -y_pred_grad
        return mu

    def _x_flatten(self, x):
        return einops.rearrange(x, '... C W H -> ... (C W H)', C=self.channels, W=self.sample_size, H=self.sample_size)

    def _x_unflatten(self, x):
        return einops.rearrange(x, '... (C W H) -> ... C W H', C=self.channels, W=self.sample_size, H=self.sample_size)

    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size

        mu = torch.zeros([batch_size, self.channels, self.sample_size, self.sample_size], device=self.device)
        
        beta = 50
        epsilon = torch.randn([self.num_sampling_steps+1, batch_size, self.channels, self.sample_size, self.sample_size], device=self.device, dtype=torch.float32)
        tempeture = 0.5
        alpha = math.exp(-tempeture * self.current_epoch)
        L_max = 10
        L_ = L_max * (1 - alpha)
        
        L = 1+int(L_)
        for l in range(L):
            
            prior = mu + epsilon[0]
            given_noise = mu[None,:] + epsilon[1:]
            
            latents = self.pipe(
                self.prompt,
                num_images_per_prompt=batch_size,
                latents=prior.type(torch.float16),
                output_type="latent",
                given_noise=given_noise,
            ).images

            gamma = math.exp(-tempeture * l)

            mu += beta * (L * gamma) * self.get_mu(latents)

        self.log_params(mu.mean(0))

        images = self.latents_to_images(latents)

        self.log_images(images)
        scores = self.get_scores(images)

        self.update_model(self._x_flatten(latents), scores)

        self.log_score(scores, stage="train")

    def log_params(self, mu):
        self.log(f"||mu||_2", mu.norm())


    def test_step(self, _, __):
        
        return

    @torch.no_grad()
    def validation_step(self, _, __):

        return


    def configure_optimizers(self):
        return torch.optim.Adam([self.dummy], lr=0.)
    
    # minimize score
    @torch.enable_grad()
    def update_model(self, x, scores):
        ''' minimize score '''

        self.data_x = torch.cat([self.data_x, x])
        self.data_y = torch.cat([self.data_y, scores])

        self.model.set_train_data(
            inputs=self.data_x, 
            targets=self.data_y, 
            strict=False,
        )