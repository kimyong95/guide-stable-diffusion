import inspect
import torch
import math

def find_closest_factors(n):
    # Start from the square root of n and move downwards to find the closest factors
    for i in range(int(math.sqrt(n)), n):
        if n % i == 0:
            return i

def disable_train(model: torch.nn.Module):
    model.eval()
    model.train = disabled_train_func
    model.requires_grad_(False)
    return model

def disabled_train_func(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class FunctionCallTracker:
    def __init__(self, func):
        self.func = func
        self.kwargs = []
        self.returns = []

    def __enter__(self):

        # Create and return the wrapped function
        def wrapper(*args, **kwargs):
            result = self.func(*args, **kwargs)
            inspect.getfullargspec(self.func).args

            func_args = inspect.getfullargspec(self.func).args
            func_args = func_args[1:] if inspect.ismethod(self.func) else func_args

            self.kwargs.append({ **dict(zip(func_args, args)), **kwargs })
            self.returns.append(result)

            return result

        return self, wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        del self.kwargs
        del self.returns
