
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
from models.diffusion_base import find_closest_factors
from palettable.colorbrewer.qualitative import Dark2_4
colors = Dark2_4.mpl_colors

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

    def __init__(self, prompt, num_sampling_steps, training_batch_size, lr, reward_func):
        super().__init__(prompt, reward_func)

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

    def derivative_y_wrt_x(self, x):

        x_flatten = self._x_flatten(x)

        y_pred = []
        y_uncertainty = []
        y_pred_grad = []
        y_pred_grad_2nd = []

        for xj in x_flatten:


            with torch.enable_grad():
                xj.requires_grad = True
                yj = self.likelihood(self.model(xj.unsqueeze(0)))
                mean = yj.mean
                # lcb = yj.mean - 2*(yj.covariance_matrix.item()**0.5)
                # loss = mean + 0.5 * lcb
                
                grad = torch.autograd.grad(mean, xj, create_graph=True, allow_unused=True)[0]
                grad_2nd = None if grad is None else \
                    torch.autograd.grad(grad.norm()*grad.shape[0], xj, create_graph=False, allow_unused=True)[0]
                
                xj.requires_grad = False
                self.model.zero_grad()

            grad = torch.zeros_like(xj) if grad is None else grad
            grad_2nd = torch.zeros_like(xj) if grad_2nd is None else grad_2nd
            
            y_pred.append(mean[0])
            y_uncertainty.append(yj.covariance_matrix[0,0])
            y_pred_grad.append(grad)
            y_pred_grad_2nd.append(grad_2nd)
        
        y_pred = torch.stack(y_pred).detach()
        y_uncertainty = torch.stack(y_uncertainty).detach()
        y_pred_grad = torch.stack(y_pred_grad).detach()
        y_pred_grad_2nd = torch.stack(y_pred_grad_2nd).detach()

        y_pred_grad = self._x_unflatten(y_pred_grad)
        y_pred_grad_2nd = self._x_unflatten(y_pred_grad_2nd)

        return y_pred_grad, y_pred_grad_2nd, y_pred, y_uncertainty

    def _x_flatten(self, x):
        return einops.rearrange(x, '... C W H -> ... (C W H)', C=self.channels, W=self.sample_size, H=self.sample_size)

    def _x_unflatten(self, x):
        return einops.rearrange(x, '... (C W H) -> ... C W H', C=self.channels, W=self.sample_size, H=self.sample_size)

    @torch.no_grad()
    def training_step(self, _, __):

        batch_size = self.training_batch_size

        mu = torch.zeros([self.num_sampling_steps+1, batch_size, self.channels, self.sample_size, self.sample_size], device=self.device)

        beta = self.lr
        epsilon = torch.randn([self.num_sampling_steps+1, batch_size, self.channels, self.sample_size, self.sample_size], device=self.device, dtype=torch.float32)

        derivative_y = torch.zeros([batch_size, self.channels, self.sample_size, self.sample_size], device=self.device, dtype=torch.float32)
        
        # ablation
        images_list = []
        scores_list = []
        outputs_list = []
        y_pred_list = []
        y_uncertainty_list = []

        L = 1 + self.current_epoch
        for i in range(L):

            prior = mu[0] + epsilon[0]
            given_noise = mu[1:,:] + epsilon[1:]

            pred_x0_traj = []
            callback = lambda i,t,pred_x0: pred_x0_traj.append(pred_x0)

            latents = self.pipe(
                self.prompt,
                num_images_per_prompt=batch_size,
                latents=prior.type(torch.float16),
                output_type="latent",
                given_noise=given_noise,
                callback=callback,
            ).images

            pred_x0_traj.append(latents)

            images = self.latents_to_images(latents)
            scores, _ = self.get_scores(images)
            images_list.append(images)
            scores_list.append(scores)

            for t in range(self.num_sampling_steps+1):
                derivative_y, derivative_y_2nd, y_pred, y_uncertainty = self.derivative_y_wrt_x(pred_x0_traj[t].type(torch.float32))
                mu[t] -= beta * derivative_y
            y_pred_list.append(y_pred)
            y_uncertainty_list.append(y_uncertainty)

        self.log_params(mu.mean(0))

        images = self.latents_to_images(latents)

        self.log_images(images)
        scores, outputs = self.get_scores(images)

        self.update_model(self._x_flatten(latents), scores)

        self.log_score(scores, stage="train")

        self.log_ablation(images_list=images_list, scores_list=scores_list, y_pred_list=y_pred_list, y_uncertainty_list=y_uncertainty_list, outputs=outputs)

    def log_ablation(self, images_list=None, scores_list=None, y_pred_list=None, y_uncertainty_list=None, outputs=None):
        path = str(self.trainer.logger.experiment.dir).removesuffix("/files") + f"/ablation/{self.current_epoch}"
        os.makedirs(path, exist_ok=True)

        ########################## plot ##########################
        if scores_list is not None and y_pred_list is not None and y_uncertainty_list is not None:

            scores = torch.stack(scores_list)
            y_pred = torch.stack(y_pred_list)
            y_uncertainty = torch.stack(y_uncertainty_list)

            scores = einops.rearrange(scores, 'L B -> B L').cpu().numpy()
            y_pred = einops.rearrange(y_pred, 'L B -> B L').cpu().numpy()
            y_uncertainty = einops.rearrange(y_uncertainty, 'L B -> B L').cpu().numpy()
            
            col = find_closest_factors(len(scores))
            row = len(scores) // col
            fig, axs = plt.subplots(row, col, figsize=(col*10, row*5))

            for i, ax in enumerate(axs.flat):

                ax.set_ylabel('scores', color=colors[0])
                ax.plot(scores[i], color=colors[0])

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
            fig.savefig(f"{path}/scores.png")

            # save tensors dict
            torch.save({
                "scores": scores,
                "y_pred": y_pred,
                "y_uncertainty": y_uncertainty,
            }, f"{path}/tensors.pt")

        ########################## plot ##########################

        ############################ save images ############################
        if images_list is not None:
            for l, images in enumerate(images_list):
                images_tensors = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
                grid_image = torchvision.utils.make_grid(images_tensors, nrow=find_closest_factors(len(images_tensors)))
                torchvision.utils.save_image(grid_image, f"{path}/l={l}.jpg", format='jpeg')
        ############################ save images ############################

        ############################ save outputs ############################
        if outputs is not None and scores_list is not None:
            final_scores = scores_list[-1]
            with open(f"{path}/llava.txt", "w") as f:
                for i, output in enumerate(outputs):
                    output = output.replace('\n', '').strip()
                    f.write(f"Score: {final_scores[i].item():3f}, LLaVA: {output}\n")
        ############################ save outputs ############################


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
