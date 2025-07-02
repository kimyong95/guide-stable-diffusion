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
import random
from contextlib import contextmanager
from torch import nn
from models.diffusion_base import find_closest_factors
from palettable.colorbrewer.qualitative import Dark2_4
colors = Dark2_4.mpl_colors

from models.guidance_models import GpGuidanceModel, NnGuidanceModel


def with_1d_support(transform_func):
    """Decorator to add 1D input support to transform methods."""
    def wrapper(self, data):
        is_1d = data.ndim == 1
        if is_1d:
            data = data.unsqueeze(0)
        
        result = transform_func(self, data)
        
        if is_1d:
            result = result.squeeze(0)
        
        return result
    return wrapper

class BaseScaler(nn.Module):
    """
    Base class for scalers. It's an "empty" scaler that does nothing.
    `fit`, `transform`, and `inverse_transform` can be overridden by subclasses.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def fit(self, data):
        """Fits the scaler to the data. For this base class, it does nothing."""
        pass

    @with_1d_support
    def transform(self, data):
        """Transforms the data. For this base class, it returns the data as is."""
        return data

    @with_1d_support
    def inverse_transform(self, data):
        """Inverse transforms the data. For this base class, it returns the data as is."""
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class StandardScaler(BaseScaler):
    def __init__(self, feature_dim):
        super().__init__(feature_dim)
        self.register_buffer('mean', torch.zeros(feature_dim))
        self.register_buffer('std', torch.ones(feature_dim))

    def fit(self, data):
        """Computes the mean and standard deviation for scaling."""
        if data.ndim == 1:
            self.mean[:] = data.mean()
            self.std[:] = data.std().clamp(min=1e-8)
        else:
            self.mean[:] = data.mean(dim=0)
            self.std[:] = data.std(dim=0).clamp(min=1e-8)

    @with_1d_support
    def transform(self, data):
        """Standardizes the data."""
        device = data.device
        return (data - self.mean.to(device)) / self.std.to(device)

    @with_1d_support
    def inverse_transform(self, data):
        """Reverts the standardization."""
        device = data.device
        return data * self.std.to(device) + self.mean.to(device)

class MinMaxNegOneZeroScaler(BaseScaler):
    def __init__(self, feature_dim):
        super().__init__(feature_dim)
        self.register_buffer('min', torch.full((feature_dim,), float('inf')))
        self.register_buffer('max', torch.full((feature_dim,), float('-inf')))
    
    def fit(self, data):
        """Computes the min and max for scaling."""
        if data.ndim == 1:
            self.min[:] = data.min()
            self.max[:] = data.max()
        else:
            self.min[:] = data.min(dim=0).values
            self.max[:] = data.max(dim=0).values

    @with_1d_support
    def transform(self, data):
        """Scales the data to the range [-1, 0]."""
        device = data.device
        range = (self.max - self.min).clamp(min=1e-8)
        return -1 + (data - self.min.to(device)) / range.to(device)

    @with_1d_support
    def inverse_transform(self, data):
        """Reverts the scaling from [-1, 0] to the original range."""
        device = data.device
        range = (self.max - self.min)
        return (data + 1) * range.to(device) + self.min.to(device)

class ExactGpModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_dim):
        super(ExactGpModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = (x_dim) ** 0.5

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ValueModel(nn.Module):
    def __init__(self, dimension, noise_level = 1e-4) -> None:
        super().__init__()
        self.dim = dimension
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_level
        self.likelihood.eval()

        model = ExactGpModel(None, None, self.likelihood, x_dim=dimension)

        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        self.x_scaler = BaseScaler(dimension)
        self.y_scaler = BaseScaler(1)

        self.all_data = {
            "x": torch.empty(0, dimension, dtype=torch.float32, device='cpu'),
            "y": torch.empty(0, dtype=torch.float32, device='cpu')
        }

    def predict(self, x):
        device = x.device
        self.model.to(device)
        self.likelihood.to(device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_preds = self.likelihood(self.model(self.x_scaler.transform(x)))
        
        y_preds_mean = self.y_scaler.inverse_transform(y_preds.mean.to(device).unsqueeze(-1)).squeeze(-1)
        y_preds_var = y_preds.variance.to(device)

        return y_preds_mean.to(device), y_preds_var.to(device)

    # x: data points
    # y: lower is better
    def add_model_data(self, x, y):
        device = x.device

        self.all_data["x"] = torch.cat([self.all_data["x"], x.cpu()], dim=0)
        self.all_data["y"] = torch.cat([self.all_data["y"], y.cpu()], dim=0)

        self.x_scaler.fit(self.all_data["x"])
        self.y_scaler.fit(self.all_data["y"])
        
        self.model.set_train_data(
            inputs=self.x_scaler.transform(self.all_data['x']).to(device),
            targets=self.y_scaler.transform(self.all_data['y']).to(device),
            strict=False
        )

    def get_model_data(self):
        return self.all_data["x"], self.all_data["y"]



class GuideDiffusionWithOptimization(DiffusionBase):

    def __init__(self, lr_mu, lr_sigma, projection, use_value_model, training_batch_size, validation_batch_size, compile, sd_model, generate_prompt, reward_query_prompt, reward_target_prompt, reward_func, max_reward_value, validation_generate_prompt):
        super().__init__(sd_model, generate_prompt, reward_func, reward_query_prompt, reward_target_prompt, max_reward_value, compile)

        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.projection = projection
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.validation_generate_prompt = validation_generate_prompt
        self.use_value_model = use_value_model

        self.dimension = self.channels * self.sample_size * self.sample_size

        self.value_model = ValueModel(self.dimension, noise_level=1e-4)

        mu = torch.zeros(self.num_sampling_steps+1, self.dimension, dtype=torch.float32, requires_grad=False)
        self.register_buffer('mu', mu)
        sigma = torch.ones(self.num_sampling_steps+1, self.dimension, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sigma', sigma) # entries is variance

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
        # scores_var: (T, B)

        assert z.shape[0] == scores.shape[0] == self.mu.shape[0] == self.sigma.shape[0]
        assert z.shape[1] == scores.shape[1]

        T_dim = z.shape[0]
        B_dim = z.shape[1]
        D_dim = self.mu.shape[1]

        
        for t in range(T_dim):
            
            # weighted sum over T
            scores_t = (scores[t:]).mean(0)
            z_t = z[t]
            scores_t_normalized = (scores_t - scores_t.mean()) / (scores_t.std() + 1e-8)
            
            scores_t_softmaxed = nn.Softmax(dim=0)(-scores_t_normalized)

            self.sigma[t] = 1 / (
                
                1/self.sigma[t] + self.lr_sigma / math.sqrt(D_dim) * (

                    (1/self.sigma[t])[None,:] * (z_t - self.mu[t,None,:]) * (z_t - self.mu[t,None,:]) * (1/self.sigma[t])[None,:] * \
                    
                    scores_t_softmaxed[:,None]
                
                # sum over B
                ).sum(0)
            )
            
            self.mu[t] = self.mu[t] - self.lr_mu / math.sqrt(D_dim) * (

                (z_t - self.mu[t][None,:]) * \
                
                scores_t_normalized[:,None]
            
            # mean over B
            ).mean(0)


    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size
        
        epsilon, epsilon_projected = self.get_noise(batch_size, self.mu, self.sigma)

        prior = epsilon_projected[0]
        given_noise = epsilon_projected[1:]

        # collect latents trajectory
        latents_trajectory = [prior]
        pred_samples_trajectory = []
        
        def callback_func(self, index, timestep, callback_kwargs):
            latents_trajectory.append(callback_kwargs["latents"])
            pred_samples_trajectory.append(callback_kwargs["pred_original_sample"])
            return {}

        generate_prompts = [self.generate_prompt for _ in range(batch_size)]

        final_latents = self.pipe(
            generate_prompts,
            latents=prior.type(torch.float16),
            output_type="latent",
            given_noise=given_noise,
            num_inference_steps=self.num_sampling_steps,
            guidance_scale=self.guidance_scale,
            callback_on_step_end=callback_func,
            callback_on_step_end_tensor_inputs=["latents", "pred_original_sample"],
        ).images
        
        final_images = self.latents_to_images(final_latents)
        final_scores, final_texts = self.get_scores(final_images)

        self.value_model.add_model_data(self._x_flatten(final_latents).to(torch.float32), final_scores)

        scores_trajectory = torch.zeros((self.num_sampling_steps+1, batch_size), dtype=torch.float32, device=self.device)
        if self.use_value_model:
            for k, pred_sample in enumerate(pred_samples_trajectory):
                scores_k, scores_var_k = self.value_model.predict(self._x_flatten(pred_sample).to(torch.float32))
                scores_trajectory[k] = scores_k
            scores_trajectory[-1] = final_scores
        else:
            scores_trajectory[:] = final_scores

        scores_trajectory = scores_trajectory

        self.log_score(scores_trajectory[-1], stage="train")
        self.update_parameters(self._x_flatten(epsilon), scores_trajectory)
        self.log_params(self.mu, self.sigma)

        self.log_ablation(images_list=[final_images], texts_list=[final_texts], scores_list=[scores_trajectory[-1]], stage="train")

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

        if self.validation_generate_prompt is None:
            prompts = [self.generate_prompt for _ in range(batch_size)]
        elif isinstance(self.validation_generate_prompt, str):
            prompts = [self.validation_generate_prompt]*batch_size
        elif isinstance(self.validation_generate_prompt, list):
            assert len(self.validation_generate_prompt) >= batch_size
            prompts = self.validation_generate_prompt[:batch_size]
            
        images = self.pipe(
            prompts,
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
