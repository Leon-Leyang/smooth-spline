import os
import sys
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from torch import nn as nn
from torchvision import transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from utils.resnet import resnet18
import numpy as np
from loguru import logger

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


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


class MLP(nn.Module):
    """
    A simple MLP for binary classification.
    """
    def __init__(self, in_features: int, out_features: int, depth: int, width: int, nonlinearity: nn.Module):
        super().__init__()
        self.register_buffer("depth", torch.as_tensor(depth))
        self.layer0 = torch.nn.Linear(in_features, width)
        for i in range(1, depth):
            setattr(
                self,
                f"layer{i}",
                nn.Linear(width, width),
            )
        self.output_layer = nn.Linear(width, out_features)
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
            logger.debug(f'Epoch {epoch}, Step {batch_idx}, Loss: {train_loss:.6f}, Accuracy: {train_accuracy:.2f}%')

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
        logger.debug(f'Test Epoch {epoch}, Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%')

        # Log the test loss and accuracy to wandb
        wandb.log({'epoch': epoch, 'val_loss': test_loss, 'val_accuracy': test_accuracy})
    else:
        logger.debug(f'Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%')

    return test_loss, test_accuracy


def get_pretrained_model(pretrained_ds='cifar100', device='cuda'):
    """
    Get the pre-trained model.
    """
    ckpt_folder = './ckpts'
    if 'cifar' in pretrained_ds:
        num_classes = 100 if 'cifar100' in pretrained_ds else 10
        model = resnet18(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{pretrained_ds}_epoch200.pth'), weights_only=True))
    elif 'mnist' in pretrained_ds:
        num_classes = 10
        model = resnet18(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{pretrained_ds}_epoch10.pth'), weights_only=True))
    elif pretrained_ds == 'imagenet':
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(device)

    return model


def fix_seed(seed=42):
    """
    Fix the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def result_exists(ds, replace_then_lp=False):
    if replace_then_lp:
        ds = f'{ds}_replace_lp'
    log_file = get_log_file_path()
    if not os.path.exists(log_file):
        return False
    with open(log_file, 'r') as f:
        for line in f:
            if f'Best accuracy for {ds}:' in line:
                return True
    return False


def set_logger(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """
    Get the logger.
    The logger will be appended to a log file if it already exists.
    """
    os.makedirs("./logs", exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=print_level)
    log_file_path = f"./logs/{name}.log"
    logger.add(
        log_file_path,
        level=logfile_level,
        mode="a"  # Append mode
    )
    return log_file_path


def get_log_file_path():
    """
    Retrieve the path of the file the logger is writing to.
    """
    file_paths = []
    for handler in logger._core.handlers.values():
        sink = handler._sink
        # Check if the sink is a file and get its path
        if hasattr(sink, "_path"):
            file_paths.append(sink._path)
    assert len(file_paths) == 1, "Only one file-based log handler is supported."
    return file_paths[0]


def plot_acc_vs_beta(acc_list, beta_list, base_acc, dataset, model_name, robust_config=None):
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
    os.makedirs('./figures', exist_ok=True)
    if robust_config:
        output_path = f"./figures/{get_file_name(get_log_file_path())}_{dataset}_{model_name}_{robust_config}.png"
    else:
        output_path = f"./figures/{get_file_name(get_log_file_path())}_{dataset}_{model_name}.png"
    plt.savefig(output_path)
