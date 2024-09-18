import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import copy
import matplotlib.pyplot as plt
from resnet import resnet18
from utils import WarmUpLR, ReplacementMapping, replace_module, get_file_name, train_epoch, test_epoch, get_data_loaders
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(mode):
    """
    Train the model on CIFAR-10.
    :param mode: mode of the train, e.g. normal/overfit
    """
    assert mode in ['normal', 'overfit'], 'Mode must be either normal or overfit'

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.1
    num_epochs = 200

    # Get the data loaders for CIFAR-10
    cifar10_train_loader, cifar10_test_loader = get_data_loaders('cifar10', batch_size)

    # Initialize the model
    model = resnet18()
    model = model.to(device)

    # Return the model if it has already been trained
    ckpt_folder = os.path.join('./ckpts', mode)
    os.makedirs(f'{ckpt_folder}', exist_ok=True)
    file_to_check = os.path.join(ckpt_folder, f'resnet18_cifar10_epoch{num_epochs}.pth')
    if os.path.exists(file_to_check):
        model.load_state_dict(torch.load(file_to_check))
        print(f'Loaded model from {file_to_check}')
        return model, cifar10_test_loader

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler with specific milestones for reduction
    if mode == 'normal':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        scheduler = None

    # Warmup scheduler
    if mode == 'normal':
        iter_per_epoch = len(cifar10_train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)
    else:
        warmup_scheduler = None

    best_test_loss = float('inf')

    # Train the model
    for epoch in range(1, num_epochs + 1):
        if epoch > 1:
            if scheduler is not None:
                scheduler.step(epoch)

        train_epoch(epoch, model, cifar10_train_loader, optimizer, criterion, device, warmup_scheduler)
        test_loss, _ = test_epoch(epoch, model, cifar10_test_loader, criterion, device)

        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_folder, f'resnet18_cifar10_epoch{epoch}.pth'))

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            print(f'Find new best model at Epoch {epoch}')
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(ckpt_folder, 'resnet18_cifar10_best.pth'))

    return model, cifar10_test_loader


def main():
    # Initialize Weights and Biases (wandb)
    wandb.init(project='smooth-spline', entity='leyang_hu')

    # Train the model on CIFAR-10
    model, cifar10_test_loader = train('normal')

    wandb.finish()


if __name__ == '__main__':
    main()
