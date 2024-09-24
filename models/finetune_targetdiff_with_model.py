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

from utils.utils import FunctionCallTracker, disable_train
from diffusers import UNet2DModel
import gpytorch
from models.diffusion_base import find_closest_factors
from palettable.colorbrewer.qualitative import Dark2_4
colors = Dark2_4.mpl_colors

from models.guidance_models import GpGuidanceModel, NnGuidanceModel, SelectBestModel
from models.finetune_with_model import FinetuneDiffusionWithModel
from models.targetdiff_base import TargetdiffBase
import asyncio

class FinetuneTargetdiffWithModel(TargetdiffBase):

    def __init__(self, guidance_model, training_batch_size, validation_batch_size, alpha, reg_mode, reg, data_id, pos_only, vina_web_url, scheduler):
        super().__init__(data_id=data_id, pos_only=pos_only, vina_web_url=vina_web_url, scheduler=scheduler)

        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size

        self.alpha = alpha
        self.reg_mode = reg_mode
        self.reg = reg

        data_x = torch.empty(0, self.num_atoms * self.dim)
        data_y = torch.empty(0)

        self.guidance_model_type = guidance_model

        if guidance_model == "gp":
            self.guidance_model = GpGuidanceModel(self.num_atoms * self.dim, noise_level=1, kernel="rbf")
            self.guidance_model._x_flatten = self._x_flatten
            self.guidance_model._x_unflatten = self._x_unflatten
        elif guidance_model == "best":
            self.guidance_model = SelectBestModel()
            self.guidance_model._x_flatten = self._x_flatten
            self.guidance_model._x_unflatten = self._x_unflatten
        else:
            raise NotImplementedError()

        self.register_buffer("data_x", data_x)
        self.register_buffer("data_y", data_y)

        # dummy parameters for pytorch lightning optimizer to work
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.automatic_optimization = False

    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size
        
        epsilon = torch.randn([self.num_sampling_steps+1, batch_size, self.num_atoms, self.dim], device=self.device, dtype=torch.float32)
        epsilon_init = epsilon.clone()

        L = 1 + self.current_epoch
        pred_pos, pred_v, epsilon = self.finetune_and_generate(epsilon, L, batch_size)
        self.log_params((epsilon-epsilon_init).mean(dim=[0,1]))

        scores, failed_count = self.get_scores_parallel(pred_pos, pred_v)

        pred_pos_tensor = torch.from_numpy(np.array(pred_pos)).to(self.device).type(torch.float32)
        self.add_data(self._x_flatten(pred_pos_tensor), scores)

        self.guidance_model.update_model_data(self.data_x, self.data_y)

        self.log_score(scores, stage="train")
        self.log(f"train/failed_rate", float(failed_count) / batch_size)
    
    @torch.no_grad()
    def finetune_and_generate(self, epsilon, L, batch_size):
        
        epsilon = epsilon.clone()
        epsilon_init = epsilon.clone()
        epsilon_init_norm = self._x_flatten(epsilon_init).norm(dim=-1)[:,:,None,None]

        # ablation
        molecules_list = []
        pred_y_list = []

        for l in range(L):
            
            pred_pos, pred_v = self.sampling(batch_size, epsilon)

            pred_pos_tensor = torch.from_numpy(np.array(pred_pos)).to(self.device).type(torch.float32)
            
            pred_y, derivative_y = self.guidance_model.derivative_y_wrt_x(pred_pos_tensor)
            pred_pos_tensor_star = pred_pos_tensor - derivative_y
            epsilon += self.alpha * (pred_pos_tensor_star - pred_pos_tensor)

            if self.reg_mode == "projection":
                epsilon_norm = self._x_flatten(epsilon).norm(dim=-1)[:,:,None,None]
                epsilon = epsilon / epsilon_norm * epsilon_init_norm
            elif self.reg_mode == "delta":
                delta = epsilon - epsilon_init
                epsilon = epsilon - self.reg * delta
            elif self.reg_mode == "delta-projection":
                delta = epsilon - epsilon_init
                epsilon = epsilon - self.reg * delta
                epsilon_norm = self._x_flatten(epsilon).norm(dim=-1)[:,:,None,None]
                epsilon = epsilon / epsilon_norm * epsilon_init_norm
            elif self.reg_mode == "pdf":
                pdf_value = torch.exp(-0.5 * (epsilon**2).sum(dim=[2,3,4]))
                epsilon = epsilon - self.reg * pdf_value[:,:,None,None] * epsilon
            elif self.reg_mode == "kl":
                epsilon = epsilon - self.reg * epsilon
            elif self.reg_mode == "none":
                pass
            else:
                raise NotImplementedError()
        
        return pred_pos, pred_v, epsilon
        
    
    def log_params(self, mu):
        self.log(f"||mu||_2", mu.norm())

    def test_step(self, _, __):
        return

    @torch.no_grad()
    def validation_step(self, _, __):

        batch_size = self.validation_batch_size

        generator = torch.Generator(device=self.device).manual_seed(1)
        epsilon = torch.randn([self.num_sampling_steps+1, batch_size, self.num_atoms, self.dim], device=self.device, dtype=torch.float32, generator=generator)
        L = 1 + self.current_epoch
        pred_pos, pred_v, epsilon = self.finetune_and_generate(epsilon, L, batch_size)

        scores, failed_count = self.get_scores_parallel(pred_pos, pred_v)
        self.log_score(scores, stage="validation")
        self.log_molecules(pred_pos, pred_v, scores)
        self.log(f"validation/failed_rate", float(failed_count) / batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam([self.dummy], lr=0.)

    def add_data(self, x, y):
        self.data_x = torch.cat([self.data_x, x])
        self.data_y = torch.cat([self.data_y, y])