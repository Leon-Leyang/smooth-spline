import torch
from torch import nn as nn


class BetaAgg(nn.Module):
    def __init__(self, beta=0, coeff=0.5, trainable=False):
        assert 0 <= beta < 1
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))
        self.beta.requires_grad_(trainable)
        self.coeff = coeff

    def forward(self, x):
        beta = self.beta
        return (self.coeff * torch.sigmoid(beta * x / (1 - beta)) * x +
                (1 - self.coeff) * torch.log(1 + torch.exp(x / (1 - beta))) * (1 - beta))


class ReplacementMapping:
    def __init__(self, beta=0.5, coeff=0.5):
        self.beta = beta
        self.coeff = coeff

    def __call__(self, module):
        if isinstance(module, torch.nn.ReLU):
            return BetaAgg(beta=self.beta, coeff=self.coeff)
        return module


def replace_module(model, beta, coeff=0.5):
    replacement_mapping = ReplacementMapping(beta=beta, coeff=coeff)

    if not isinstance(model, torch.nn.Module):
        raise ValueError("Torch.nn.Module expected as input")
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if name == "":
            continue
        replacement = replacement_mapping(module).to(device)
        module_names = name.split(".")
        # we go down the tree up to the parent
        parent = model
        for name in module_names[:-1]:
            parent = getattr(parent, name)
        setattr(parent, module_names[-1], replacement)
    return model
