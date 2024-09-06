import torch
import torch.nn as nn


def replacement(name, module):
    if isinstance(module, torch.nn.ReLU):
        # TODO: return your own smooth verions
        return
    return module

def replace_module(model, replacement_mapping):
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Torch.nn.Module expected as input")
    for name, module in model.named_modules():
        if name == "":
            continue
        replacement = replacement_mapping(name, module)
        module_names = name.split(".")
        # we go down the tree up to the parent
        parent = model
        for name in module_names[:-1]:
            parent = getattr(parent, name)
        setattr(parent, module_names[-1], replacement)
    return model


class SmoothReLU(nn.Module):
    def __init__(self, smoothness=0,in_features=1, trainable=False):
        super().__init__()
        param = torch.nn.Parameter(torch.zeros(in_features)+smoothness)
        param.requires_grad_(trainable)
        self.register_parameter("smoothness", param)

    def forward(self, x):
        return torch.sigmoid(x / nn.functional.softplus(self.smoothness)) * x

class LazySmoothReLU(nn.modules.lazy.LazyModuleMixin, SmoothReLU):

    cls_to_become = SmoothReLU
    weight: nn.parameter.UninitializedParameter

    def __init__(self, axis=-1, device=None, dtype=None):
        super().__init__()
        if type(axis) not in [tuple, list]:
            axis = [axis]
        self.axis = axis
        self.smoothness = nn.parameter.UninitializedParameter(device=device, dtype=dtype)

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                s = [1 for _ in range(input.ndim)]
                for i in self.axis:
                    s[i] = input.size(i)
                self.smoothness.materialize(s)
                self.smoothness.copy_(torch.Tensor([0.541323]))
