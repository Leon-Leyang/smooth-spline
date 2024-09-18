import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms

import wandb
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


def train_epoch(epoch, model, trainloader, optimizer, criterion, device, warmup_scheduler):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            train_loss = running_loss / (batch_idx + 1)
            train_accuracy = 100. * correct / total
            print(f'Epoch {epoch}, Step {batch_idx}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        if epoch <= 1:
            if warmup_scheduler is not None:
                warmup_scheduler.step()

    # Log the training loss and accuracy to wandb
    wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'lr': optimizer.param_groups[0]['lr']})


def test_epoch(epoch, model, testloader, criterion, device):
    """
    Test the model for one epoch.
    Specify epoch=-1 to use for testing after training.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader)
    test_accuracy = 100. * correct / total
    if epoch != -1:
        print(f'Test Epoch {epoch}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

        # Log the test loss and accuracy to wandb
        wandb.log({'epoch': epoch, 'val_loss': test_loss, 'val_accuracy': test_accuracy})
    else:
        print(f'Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    return test_loss, test_accuracy


def get_data_loaders(dataset, batch_size=128):
    """
    Get the data loaders for the dataset.
    """
    assert dataset in ['cifar10', 'cifar100']

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainloader, testloader
