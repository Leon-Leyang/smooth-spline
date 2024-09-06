import torch


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