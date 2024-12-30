import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms as transforms


# Predefined normalization values for different datasets
NORMALIZATION_VALUES = {
    'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    'mnist': ([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
}


def replicate_if_needed(x):
    """
    Replicate a single-channel tensor to three channels if needed.
    If the input has more than one channel, it is returned as is.
    """
    if x.shape[0] == 1:  # Check if there's only one channel
        return x.repeat(3, 1, 1)  # Replicate to 3 channels
    return x  # Return unchanged if already has more than 1 channel


def get_data_loaders(dataset, train_batch_size=500, test_batch_size=500, train_size=None, num_workers=6):
    """
    Get the data loaders for the dataset.
    """
    if '_to_' in dataset:  # e.g., cifar10_to_cifar100
        transform_to_use = dataset.split('_to_')[0]
    else:
        transform_to_use = dataset

    if '_to_' in dataset:
        dataset_to_use = dataset.split('_to_')[-1]
    else:
        dataset_to_use = dataset

    normalization_to_use = dataset

    if 'cifar10' in transform_to_use:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
        ])
    elif 'cifar100' in transform_to_use:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
        ])
    elif 'mnist' in transform_to_use:
        transform_train = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
        ])
    elif 'imagenet' in transform_to_use:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),
            transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(replicate_if_needed),  # Apply conditional replication
            transforms.Normalize(*NORMALIZATION_VALUES[normalization_to_use])
        ])

    if dataset_to_use == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset_to_use == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset_to_use == 'mnist':
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset_to_use == 'imagenet':
        trainset = None
        testset = torchvision.datasets.ImageNet(
            root='./data/imagenet', split='val', transform=transform_test)
    else:
        raise NotImplementedError(f'The specified dataset {dataset_to_use} is not implemented.')

    if train_size is not None:
        indices = np.random.choice(len(trainset), train_size, replace=False)
        trainset = Subset(trainset, indices)

    if dataset_to_use != 'imagenet':
       trainloader = torch.utils.data.DataLoader(
           trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    else:
        trainloader = None
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader
