import torch
from torch import nn as nn


class BetaSwish(nn.Module):
    def __init__(self, beta=0, in_features=1, trainable=False):
        assert 0 <= beta < 1
        super().__init__()
        param = torch.nn.Parameter(torch.zeros(in_features)+beta)
        param.requires_grad_(trainable)
        self.register_parameter("beta", param)

    def forward(self, x):
        return torch.sigmoid(self.beta * x / (1 - self.beta)) * x


class LazyBetaSwish(nn.modules.lazy.LazyModuleMixin, BetaSwish):

    cls_to_become = BetaSwish
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
                self.beta.materialize(s)
                self.beta.copy_(torch.Tensor([self.val_beta]))


class BetaAgg(nn.Module):
    def __init__(self, beta=0, in_features=1, trainable=False, coeff=0.5):
        assert 0 <= beta < 1
        super().__init__()
        param = torch.nn.Parameter(torch.zeros(in_features)+beta)
        param.requires_grad_(trainable)
        self.register_parameter("beta", param)
        self.coeff = coeff

    def forward(self, x):
        return (self.coeff * torch.sigmoid(self.beta * x / (1 - self.beta)) * x +
                (1 - self.coeff) * nn.functional.softplus(x, beta=self.beta))


class LazyBetaAgg(nn.modules.lazy.LazyModuleMixin, BetaAgg):
    cls_to_become = BetaAgg
    weight: nn.parameter.UninitializedParameter

    def __init__(self, axis=-1, beta=0.5, coeff=0.5, device=None, dtype=None):
        super().__init__()
        if type(axis) not in [tuple, list]:
            axis = [axis]
        self.axis = axis
        self.val_beta = beta
        self.beta = nn.parameter.UninitializedParameter(device=device, dtype=dtype)
        self.coeff = coeff

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                # Determine shape for the beta parameter
                s = [1 for _ in range(input.ndim)]
                self.beta.materialize(s)
                self.beta.copy_(torch.Tensor([self.val_beta]))


