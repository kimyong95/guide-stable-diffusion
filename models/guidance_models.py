import gpytorch
import torch
from utils.utils import disable_train
from torch import nn

class ExactGpModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_dim, kernel="rbf"):
        super(ExactGpModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=x_dim))
        elif kernel == "linear":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        else:
            raise NotImplementedError()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GpGuidanceModel(nn.Module):
    def __init__(self, dimension, kernel="rbf") -> None:
        super().__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = 1e-4
        self.likelihood.eval()

        model = ExactGpModel(None, None, self.likelihood, x_dim=dimension, kernel=kernel)
        if kernel == "rbf":
            model.covar_module.base_kernel.lengthscale = (dimension) ** 0.5

        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        self._x_flatten = None
        self._x_unflatten = None
    
    def derivative_y_wrt_x(self, x):
        
        device = x.device

        if self.model.train_inputs == None or len(self.model.train_inputs) == 0:
            return torch.zeros(x.shape[0], device=device), torch.zeros_like(x)

        x_flatten = self._x_flatten(x)

        y_pred = []
        y_grad = []

        for xj in x_flatten:

            with torch.enable_grad():
                xj.requires_grad = True

                yj = self.likelihood(self.model(xj.unsqueeze(0)))

                mean = yj.mean
                lcb = yj.mean - 2*(yj.covariance_matrix.item()**0.5)
                loss = mean + 0.5 * lcb

                grad = torch.autograd.grad(mean, xj, create_graph=False, allow_unused=True)[0]

                xj.requires_grad = False

            y_pred.append(mean.detach())
            y_grad.append(grad.detach())

        y_pred = torch.stack(y_pred)
        y_grad = torch.stack(y_grad)

        y_grad = self._x_unflatten(y_grad)

        return y_pred, y_grad

    @torch.enable_grad()
    def update_model_data(self, x, y):
        self.model.set_train_data(
            inputs=x, 
            targets=y,
            strict=False,
        )



from torch import nn
import torch.nn.functional as F

class CnnRegresor(nn.Module):
    def __init__(self, input_channels=4, input_size=128):
        super().__init__()

        linear_input_size = input_size * 8
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(in_features=128 * (input_size // 8) * (input_size // 8), out_features=linear_input_size)
        self.fc2 = nn.Linear(in_features=linear_input_size, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First convolution layer + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Second convolution layer + ReLU + pooling
        x = self.pool(F.relu(self.conv3(x)))  # Third convolution layer + ReLU + pooling
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))    # First fully connected layer + ReLU
        x = F.relu(self.fc2(x))    # Second fully connected layer + ReLU
        x = -torch.sigmoid(self.fc3(x)).squeeze()  # Output layer with sigmoid activation for output range [-1, 0]
        
        return x

class NnGuidanceModel(nn.Module):

    def __init__(self, input_channels=4, input_size=128, batch_size=32):
        super().__init__()

        self.model = CnnRegresor(input_channels=input_channels, input_size=input_size)
        self.eval()

        self._x_flatten = None
        self._x_unflatten = None
        self._trained = False
        self._batch_size = batch_size

    def derivative_y_wrt_x(self, x):

        device = x.device

        if not self._trained:
            return torch.zeros(x.shape[0], device=device), torch.zeros_like(x)

        x_flatten = self._x_flatten(x)

        y_pred = []
        y_grad = []

        for xj in x_flatten:

            with torch.enable_grad():
                xj.requires_grad = True

                yj = self.model(self._x_unflatten(xj.unsqueeze(0)))

                grad = torch.autograd.grad(yj, xj, create_graph=False, allow_unused=True)[0]

                xj.requires_grad = False

            y_pred.append(yj.detach())
            y_grad.append(grad.detach())

        y_pred = torch.stack(y_pred)
        y_grad = torch.stack(y_grad)

        y_grad = self._x_unflatten(y_grad)

        return y_pred, y_grad

    @torch.enable_grad()
    def update_model_data(self, x, y):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        
        for _ in range(100):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred_y = self.model(self._x_unflatten(batch_x))
                loss = F.mse_loss(pred_y, batch_y)
                loss.backward()
                optimizer.step()
        self.model.eval()

        self._trained = True