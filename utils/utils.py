import inspect
import torch

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
