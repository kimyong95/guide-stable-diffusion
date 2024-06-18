
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
import cma
from utils.finetune_difussers import FinetuneStableDiffusionPipeline, FinetuneDPMSolverMultistepScheduler

class CMA(DiffusionBase):

    def __init__(self, prompt, num_sampling_steps, training_batch_size):
        super().__init__(prompt)

        self.num_sampling_steps = num_sampling_steps
        self.training_batch_size = training_batch_size

        self.es = None

        # dummy parameters for pytorch lightning optimizer to work
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.automatic_optimization = False

    def _x_flatten(self, x):
        return einops.rearrange(x, '... C W H -> ... (C W H)', C=self.channels, W=self.sample_size, H=self.sample_size)

    def _x_unflatten(self, x):
        return einops.rearrange(x, '... (C W H) -> ... C W H', C=self.channels, W=self.sample_size, H=self.sample_size)

    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size

        latents = self.es.ask()
        latents = torch.from_numpy(np.stack(latents).astype(np.float16)).to(self.device)
        
        images = self.latents_to_images(self._x_unflatten(latents))
        self.log_images(images)

        scores = self.get_scores(images)
        
        self.es.tell(latents.cpu().numpy(), scores.cpu().numpy())

        self.log_score(scores, stage="train")
        self.log_params()

        # dummy
        opt = self.optimizers()
        opt.step()
    
    def on_fit_start(self):
        super().on_fit_start()

        batch_size = self.training_batch_size

        latents = self.pipe(
            self.prompt,
            num_images_per_prompt=batch_size,
            output_type="latent",
        ).images

        images = self.latents_to_images(latents)
        scores = self.get_scores(images)

        init_x = latents[torch.sort(scores).indices[len(scores)//2].item()]

        self.es = cma.CMAEvolutionStrategy(
            self._x_flatten(init_x).cpu().numpy(),
            0.1,
            inopts={
                "popsize": batch_size,
            },
        )

        return self

    def test_step(self, _, __):
        
        return

    @torch.no_grad()
    def validation_step(self, _, __):

        return

    def log_params(self):

        self.log(f"||mu||_2", torch.from_numpy(self.es.mean).norm())
        
        sigma = self.es.sigma * torch.from_numpy(self.es.C)
        max_eig = sigma.max()
        min_eig = sigma.min()

        self.log(f"sigma max eigen", max_eig)
        self.log(f"sigma min eigen", min_eig)

    def configure_optimizers(self):
        return torch.optim.Adam([self.dummy], lr=0.)