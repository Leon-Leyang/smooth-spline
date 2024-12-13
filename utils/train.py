import os
import torch
import wandb
from torch import nn as nn, optim as optim
from utils.resnet import resnet18
from utils.utils import WarmUpLR, train_epoch, test_epoch, set_logger, get_file_name
from loguru import logger
from utils.data import get_data_loaders
import argparse


def train(dataset):
    """
    Train the model on the specified dataset.
    :param dataset: dataset to train on, e.g. cifar10/cifar100
    """
    logger.info(f'Training ResNet18 on {dataset}...')
    wandb.init(project='smooth-spline', entity='leyang_hu')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.1
    num_epochs = 200 if dataset != 'mnist' else 10

    # Get the data loaders
    train_loader, test_loader = get_data_loaders(dataset, train_batch_size=batch_size)

    # Initialize the model
    num_classes = 100 if 'cifar100' in dataset else 10
    model = resnet18(num_classes=num_classes)
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

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_folder, f'resnet18_{dataset}_epoch{epoch}.pth'))

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            logger.debug(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(ckpt_folder, f'resnet18_{dataset}_best.pth'))

    wandb.finish()
    logger.info(f'Finished training!')
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Train a model on the specified dataset.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to train on, e.g. cifar10/cifar100')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_{args.dataset}')
    train(args.dataset)
