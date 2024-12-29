import torch
import torch.nn as nn
from utils.resnet import resnet18


class HookedModel(nn.Module):
    def __init__(self, model, topk=1):
        super().__init__()
        self.model = model
        self.topk = topk
        self._features = {}

        chosen_layers = get_topk_layers(model, topk)

        for name, module in self.model.named_children():
            if module in chosen_layers:
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            self._features[layer_name] = output

        return hook

    def forward(self, x):
        self.model(x)
        feats_cat = []
        for k, feat in self._features.items():
            feats_cat.append(feat.flatten(1))
        feats_cat = torch.cat(feats_cat, dim=1) if feats_cat else None

        return feats_cat


def get_topk_layers(model, topk):
    # Get the top-level children
    children = list(model.children())

    # Exclude the last one if it's the classification head:
    if isinstance(children[-1], nn.Linear):
        children = children[:-1]

    # Now just take the last `topk` from that shortened list
    if topk > len(children):
        topk = len(children)
    return children[-topk:]


model = resnet18(num_classes=100)
hooked_model = HookedModel(model, topk=3)

# Test the forward pass
x = torch.randn(10, 3, 224, 224)
feats_cat = hooked_model(x)
print(feats_cat.shape)

x = torch.randn(10, 3, 224, 224)
feats_cat = hooked_model(x)
print(feats_cat.shape)
