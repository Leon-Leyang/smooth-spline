import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


class ReplacementMapping:
    def __init__(self, beta=0.5):
        self.beta = beta

    def __call__(self, name, module):
        if isinstance(module, torch.nn.ReLU):
            return LazyBetaReLU(beta=self.beta)
        return module


def replace_module(model, replacement_mapping):
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Torch.nn.Module expected as input")
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if name == "":
            continue
        replacement = replacement_mapping(name, module).to(device)
        module_names = name.split(".")
        # we go down the tree up to the parent
        parent = model
        for name in module_names[:-1]:
            parent = getattr(parent, name)
        setattr(parent, module_names[-1], replacement)
    return model


class SmoothReLU(nn.Module):
    def __init__(self, smoothness=0, in_features=1, trainable=False):
        super().__init__()
        param = torch.nn.Parameter(torch.zeros(in_features)+smoothness)
        param.requires_grad_(trainable)
        self.register_parameter("smoothness", param)

    def forward(self, x):
        return torch.sigmoid(x / nn.functional.softplus(self.smoothness)) * x


class LazySmoothReLU(nn.modules.lazy.LazyModuleMixin, SmoothReLU):

    cls_to_become = SmoothReLU
    weight: nn.parameter.UninitializedParameter

    def __init__(self, axis=-1, smoothness=0.541323, device=None, dtype=None):
        super().__init__()
        if type(axis) not in [tuple, list]:
            axis = [axis]
        self.axis = axis
        self.val_smoothness = smoothness
        self.smoothness = nn.parameter.UninitializedParameter(device=device, dtype=dtype)

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                s = [1 for _ in range(input.ndim)]
                for i in self.axis:
                    s[i] = input.size(i)
                self.smoothness.materialize(s)
                self.smoothness.copy_(torch.Tensor([self.val_smoothness]))


class BetaReLU(nn.Module):
    def __init__(self, beta=0, in_features=1, trainable=False):
        assert 0 <= beta < 1
        super().__init__()
        param = torch.nn.Parameter(torch.zeros(in_features)+beta)
        param.requires_grad_(trainable)
        self.register_parameter("beta", param)

    def forward(self, x):
        return torch.sigmoid(self.beta * x / (1 - self.beta)) * x


class LazyBetaReLU(nn.modules.lazy.LazyModuleMixin, BetaReLU):

    cls_to_become = BetaReLU
    weight: nn.parameter.UninitializedParameter

    def __init__(self, axis=-1, beta=0.5, device=None, dtype=None):
        super().__init__()
        if type(axis) not in [tuple, list]:
            axis = [axis]
        self.axis = axis
        self.val_beta = beta
        self.beta = nn.parameter.UninitializedParameter(device=device, dtype=dtype)

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                s = [1 for _ in range(input.ndim)]
                for i in self.axis:
                    s[i] = input.size(i)
                self.beta.materialize(s)
                self.beta.copy_(torch.Tensor([self.val_beta]))


class Network(nn.Module):
    """
    A simple neural network for binary classification.
    """
    def __init__(self, in_features: int, depth: int, width: int, nonlinearity: nn.Module):
        super().__init__()
        self.register_buffer("depth", torch.as_tensor(depth))
        self.layer0 = torch.nn.Linear(in_features, width)
        for i in range(1, depth):
            setattr(
                self,
                f"layer{i}",
                nn.Linear(width, width),
            )
        self.output_layer = nn.Linear(width, 1)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        for i in range(self.depth):
            x = getattr(self, f"layer{i}")(x)
            x = self.nonlinearity(x)
        x = self.output_layer(x)
        return x


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_file_name(calling_file):
    """
    Returns the file name of the calling file without the extension.
    """
    file_name = os.path.basename(calling_file)
    return os.path.splitext(file_name)[0]
