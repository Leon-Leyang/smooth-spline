import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import wandb
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from torchvision import transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Subset
from robustbench.eval import benchmark


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


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
        total_iters: total_iters of warmup phase
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
            print(f'Epoch {epoch}, Step {batch_idx}, Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.2f}%')

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
        print(f'Test Epoch {epoch}, Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%')

        # Log the test loss and accuracy to wandb
        wandb.log({'epoch': epoch, 'val_loss': test_loss, 'val_accuracy': test_accuracy})
    else:
        print(f'Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%')

    return test_loss, test_accuracy


class GMMNoiseAdder:
    def __init__(self, gaussians, num_classes=10, alpha=0.001):
        """
        :param gaussians: a list of tuples where each tuple contains the mean and standard deviation of a Gaussian
        :param num_classes: the number of classes in the dataset
        :param alpha: the dominance factor for the selected Gaussian
        """
        self.gaussians = gaussians
        self.num_classes = num_classes
        self.alpha = alpha

        # Precompute weights for each class
        self.class_weights = self._generate_class_weights()

    def _generate_class_weights(self):
        """
        Generates weights for each class where one Gaussian has a higher weight.
        """
        class_weights = {}
        base_weight = 1 / self.num_classes
        for class_idx in range(self.num_classes):
            weights = [base_weight] * self.num_classes
            for idx, weight in enumerate(weights):
                if idx == class_idx:
                    weights[idx] = base_weight + self.alpha
                else:
                    weights[idx] = base_weight - self.alpha / (self.num_classes - 1)
            class_weights[class_idx] = weights
        return class_weights

    def __call__(self, img, target):
        """
        img: the image tensor
        target: the class label (int)
        """
        # Get the weights for this class
        weights = self.class_weights[target]

        # Sample a Gaussian index based on the class-specific weights
        gaussian_idx = np.random.choice(self.num_classes, p=weights)
        mean, std = self.gaussians[gaussian_idx]

        # Apply noise from the selected Gaussian
        noise = mean + torch.randn_like(img) * std
        noisy_img = img + noise
        return noisy_img


class NoisyCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, noise_adder=None, **kwargs):
        super(NoisyCIFAR10, self).__init__(*args, **kwargs)
        self.noise_adder = noise_adder

    def __getitem__(self, index):
        img, target = super(NoisyCIFAR10, self).__getitem__(index)
        if self.noise_adder is not None:
            img = self.noise_adder(img, target)
        return img, target


def get_data_loaders(dataset, train_batch_size=128, test_batch_size=2000, mode='normal'):
    """
    Get the data loaders for the dataset.
    """
    assert dataset in ['cifar10', 'cifar100', 'noisy_cifar10', 'noisy_cifar100'], 'Dataset must be either cifar10, cifar100, noisy_cifar10 or noisy_cifar100'

    if dataset == 'noisy_cifar10':
        gaussians = [(i * 0.01, 0) for i in range(10)]
        noise_adder = GMMNoiseAdder(gaussians, num_classes=10, alpha=0.8)
    elif dataset == 'noisy_cifar100':
        raise NotImplementedError('Noisy CIFAR-100 is not implemented yet.')

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
    elif dataset == 'noisy_cifar10':
        trainset = NoisyCIFAR10(
            root='./data', train=True, download=True, transform=transform_train, noise_adder=noise_adder)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'noisy_cifar100':
        raise NotImplementedError('Noisy CIFAR-100 is not implemented yet.')

    if mode == 'overfit':
        indices = np.random.choice(len(trainset), 2000, replace=False)
        trainset = Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=8)
    return trainloader, testloader


def replace_and_test_acc(model, test_loader, beta_vals, mode, dataset, calling_file):
    """
    Replace ReLU with BetaReLU and test the model on the specified dataset.
    """
    assert mode in ['normal', 'suboptimal', 'overfit'], 'Mode must be either normal, suboptimal or overfit'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('*' * 50)
    print(f'Running post-replace accuracy test on {dataset}...')
    print('*' * 50)
    criterion = nn.CrossEntropyLoss()

    acc_list = []
    beta_list = []

    # Test the original model
    print('Testing the original model...')
    _, base_acc = test_epoch(-1, model, test_loader, criterion, device)
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        print(f'Using BetaReLU with beta={beta:.3f}')
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        _, test_acc = test_epoch(-1, new_model, test_loader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)
    acc_list.append(base_acc)
    beta_list.append(1)
    print(f'Best accuracy: {best_acc:.2f} with beta={best_beta:.3f}, compared to ReLU accuracy: {base_acc:.2f}')

    # Plot the test accuracy vs beta values
    plt.figure(figsize=(12, 8))
    plt.plot(beta_list, acc_list)
    plt.axhline(y=base_acc, color='r', linestyle='--', label='ReLU Test Accuracy')
    plt.xlabel('Beta')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Beta Values')

    # Ensure that both x-axis and y-axis show raw numbers without offset or scientific notation
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    plt.xticks(beta_list[::5], rotation=45)
    plt.legend()
    output_folder = os.path.join("./figures", get_file_name(calling_file))
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"replace_and_test_acc_{dataset}_{mode}.png"))
    plt.show()


def replace_and_test_robustness(model, threat, beta_vals, mode, dataset, calling_file, batch_size=2000, n_examples=1000,
                                transform_test=DEFAULT_TRANSFORM):
    """
    Replace ReLU with BetaReLU and test the model's robustness on RobustBench.
    """
    assert mode in ['normal', 'suboptimal', 'overfit'], 'Mode must be either normal, suboptimal or overfit'

    threat_to_eps = {
        'Linf': 8 / 255,
        'L2': 0.5
    }

    model.eval()
    model_name = model.__class__.__name__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'cifar100_to_cifar10' in dataset:
        dataset_to_use = 'cifar10'
    elif 'noisy_cifar10' in dataset:
        dataset_to_use = 'cifar10'
    else:
        dataset_to_use = dataset

    print('*' * 50)
    print(f'Running post-replace robustness test on {dataset_to_use}...')
    print('*' * 50)

    robust_acc_list = []
    beta_list = []

    state_path_format_str = f"./cache/{model_name}_{dataset}_{mode}_{threat}_{{beta:.2f}}.json"

    # Test the original model
    print('Testing the original model...')
    state_path = Path(state_path_format_str.format(beta=1))
    _, base_robust_acc = benchmark(
        model, dataset=dataset_to_use, threat_model=threat, eps=threat_to_eps[threat], device=device,
        batch_size=batch_size, preprocessing=transform_test, n_examples=n_examples, aa_state_path=state_path
    )
    base_robust_acc *= 100
    best_robust_acc = base_robust_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        print(f'Using BetaReLU with beta={beta:.2f}')
        state_path = Path(state_path_format_str.format(beta=beta))
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        _, robust_acc = benchmark(
            new_model, dataset=dataset_to_use, threat_model=threat, eps=threat_to_eps[threat], device=device,
            batch_size=batch_size, preprocessing=transform_test, n_examples=n_examples, aa_state_path=state_path
        )
        robust_acc *= 100
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            best_beta = beta
        robust_acc_list.append(robust_acc)
        beta_list.append(beta)
    robust_acc_list.append(base_robust_acc)
    beta_list.append(1)
    print(f'Best robust accuracy: {best_robust_acc:.2f} with beta={best_beta:.2f}, compared to ReLU accuracy: {base_robust_acc:.2f}')

    # Plot the test accuracy vs beta values
    plt.figure(figsize=(12, 8))
    plt.plot(beta_list, robust_acc_list)
    plt.axhline(y=base_robust_acc, color='r', linestyle='--', label='ReLU Robust Accuracy')
    plt.xlabel('Beta')
    plt.ylabel('Robust Accuracy')
    plt.title('Robust Accuracy vs Beta Values')

    # Ensure that both x-axis and y-axis show raw numbers without offset or scientific notation
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    plt.xticks(beta_list[::5], rotation=45)
    plt.legend()
    output_folder = os.path.join("./figures", get_file_name(calling_file))
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"replace_and_test_robustness_{model_name}_{dataset}_{mode}_{threat}.png"))
    plt.show()
