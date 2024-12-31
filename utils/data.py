import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms as transforms


# Predefined normalization values for different datasets
NORMALIZATION_VALUES = {
    'cifar10': ([0.491, 0.482, 0.447], [0.247, 0.244, 0.262]),
    'cifar100': ([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
    'mnist': ([0.131, 0.131, 0.131], [0.308, 0.308, 0.308]),
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'arabic_characters': ([0.101, 0.101, 0.101], [0.301, 0.301, 0.301]),
    'fgvc_aircraft': ([0.485, 0.52, 0.548], [0.219, 0.21, 0.241]),
    'places365_small': ([0.458, 0.441, 0.408], [0.269, 0.267, 0.285]),
    'flowers102': ([0.43, 0.38, 0.295], [0.295, 0.246, 0.273]),
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
        dataset_to_use = dataset.split('_to_')[-1]
        normalization_to_use = dataset.split('_to_')[-1]
    else:
        transform_to_use = dataset
        dataset_to_use = dataset
        normalization_to_use = dataset

    if transform_to_use == 'cifar10':
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
    elif transform_to_use == 'cifar100':
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
    elif transform_to_use == 'mnist':
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
    elif transform_to_use == 'imagenet':
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
