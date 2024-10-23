import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import transforms as transforms


class NoisyCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, noise_adder=None, **kwargs):
        super(NoisyCIFAR10, self).__init__(*args, **kwargs)
        self.noise_adder = noise_adder

    def __getitem__(self, index):
        img, target = super(NoisyCIFAR10, self).__getitem__(index)
        if self.noise_adder is not None:
            img = self.noise_adder(img, target)
        return img, target


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


def get_data_loaders(dataset, train_batch_size=128, test_batch_size=2000, train_size=None):
    """
    Get the data loaders for the dataset.
    """
    if 'cifar' in dataset:
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
    elif dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset == 'imagenet':
        transform_train = None
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        gaussians = [(i * 0.01, 0) for i in range(10)]
        noise_adder = GMMNoiseAdder(gaussians, num_classes=10, alpha=0.8)
        trainset = NoisyCIFAR10(
            root='./data', train=True, download=True, transform=transform_train, noise_adder=noise_adder)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'imagenet':
        trainset = None
        testset = torchvision.datasets.ImageNet(
            root='./data/imagenet', split='val', transform=transform_test)
    else:
        raise NotImplementedError(f'The specified dataset {dataset} is not implemented.')

    if train_size is not None:
        indices = np.random.choice(len(trainset), train_size, replace=False)
        trainset = Subset(trainset, indices)

    if dataset != 'imagenet':
       trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    else:
        trainloader = None
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=8)
    return trainloader, testloader


if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders('mnist')
