import os
import wandb
from torch import optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from utils.model import *
from utils.utils import set_logger, get_file_name
from loguru import logger
from utils.data import get_data_loaders, replicate_if_needed, NORMALIZATION_VALUES
import argparse
from torchvision import transforms as transforms


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


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimizer (e.g., SGD)
        total_iters: total_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Set the learning rate to base_lr * epoch / total_iters"""
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def train(dataset, model_name):
    """
    Train the model on the specified dataset.
    :param dataset: dataset to train on, e.g., cifar10/cifar100
    """
    name_to_model = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'vgg19': vgg19
    }

    logger.info(f'Training {model_name} on {dataset}...')
    wandb.init(project='smooth-spline', entity='')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.1
    num_epochs = 200 if dataset != 'mnist' else 10

    # Get the data loaders
    transform_train, transform_test = None, None
    if model_name == 'vgg19' and dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize(32),  # Hack for avoiding too small images for VGG
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES['mnist'])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),  # Hack for avoiding too small images for VGG
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES['mnist'])
        ])

    train_loader, test_loader = get_data_loaders(dataset, train_batch_size=batch_size, transform_train=transform_train, transform_test=transform_test)

    # Initialize the model
    num_classes = 100 if 'cifar100' in dataset else 10

    model = name_to_model[model_name](num_classes=num_classes)
    model = model.to(device)

    # Create the checkpoint folder
    ckpt_folder = './ckpts'
    os.makedirs(f'{ckpt_folder}', exist_ok=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler with specific milestones for reduction
    if dataset != 'mnist':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.2)

    # Warmup scheduler
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    best_test_loss = float('inf')

    # Train the model
    for epoch in range(1, num_epochs + 1):
        if epoch > 1:
            if scheduler is not None:
                scheduler.step(epoch)

        train_epoch(epoch, model, train_loader, optimizer, criterion, device, warmup_scheduler)
        test_loss, _ = test_epoch(epoch, model, test_loader, criterion, device)

        # Save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_folder, f'{model_name}_{dataset}_epoch{epoch}.pth'))

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            logger.debug(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(ckpt_folder, f'{model_name}_{dataset}_best.pth'))

    wandb.finish()
    logger.info(f'Finished training!')
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Train a model on the specified dataset.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to train on, e.g., cifar10/cifar100')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to train, e.g., resnet18/vgg16')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_{args.dataset}_{args.model}')
    train(args.dataset, args.model)
