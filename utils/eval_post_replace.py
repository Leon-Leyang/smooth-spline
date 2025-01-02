import copy
import os
from pathlib import Path
import torch
from torch import nn as nn
from torchvision import transforms as transforms
from utils.robustbench import benchmark
from utils.utils import plot_acc_vs_beta
from utils.smooth_spline import replace_module
from train import test_epoch
from loguru import logger
from utils.data import get_data_loaders


def replace_and_test_acc(model, beta_vals, dataset, coeff=0.5):
    """
    Replace ReLU with BetaReLU and test the model on the specified dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = model.base_model.__class__.__name__ if hasattr(model, 'base_model') \
        else model.feature_extractor.base_model.__class__.__name__ if hasattr(model, 'feature_extractor') \
        else model.__class__.__name__

    _, test_loader = get_data_loaders(dataset)

    logger.info(f'Running post-replace accuracy test for {model_name} on {dataset}...')
    criterion = nn.CrossEntropyLoss()

    acc_list = []
    beta_list = []

    # Test the original model
    logger.debug('Using ReLU...')
    _, base_acc = test_epoch(-1, model, test_loader, criterion, device)
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using BetaReLU with beta={beta:.3f}')
        new_model = replace_module(copy.deepcopy(model), beta, coeff=coeff)

        # Register the hook for the top-k layer as copy.deepcopy does not copy hooks
        if hasattr(model, 'feature_extractor'):
            new_model.feature_extractor.register_hook(new_model.feature_extractor.topk)

        _, test_acc = test_epoch(-1, new_model, test_loader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)
    acc_list.append(base_acc)
    beta_list.append(1)
    logger.info(f'Best accuracy for {dataset}: {best_acc:.2f} with beta={best_beta:.2f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_acc_vs_beta(acc_list, beta_list, base_acc, dataset, model_name)


def replace_and_test_robustness(model, threat, beta_vals, dataset, coeff=0.5, seed=42, batch_size=2000, n_examples=10000, model_id=None):
    """
    Replace ReLU with BetaReLU and test the model's robustness on RobustBench.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model_name = model_id if model_id is not None \
        else model.base_model.__class__.__name__ if hasattr(model, 'base_model') \
        else model.__class__.__name__

    threat_to_eps = {
        'Linf': 8 / 255,
        'L2': 0.5,
        'corruptions': None,
    }

    dataset_to_transform = {
        'cifar10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.491, 0.482, 0.447], [0.247, 0.244, 0.262]),
        ]),
        'cifar100': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]),
        ]),
        'imagenet': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    if threat == 'corruptions':     # No need to resize and crop for ImageNet-C as it is already 224x224
        dataset_to_transform['imagenet'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    logger.info(f'Running post-replace robustness test for {model_name} on {dataset} with {threat} attack...')

    acc_list = []
    beta_list = []

    os.makedirs('./cache', exist_ok=True)
    state_path_format_str = f"./cache/{model_name}_{dataset}_{threat}_{n_examples}_{{beta:.2f}}.json"

    # Test the original model
    logger.debug('Using ReLU...')
    state_path = Path(state_path_format_str.format(beta=1))
    _, base_acc = benchmark(
        model, dataset=dataset, threat_model=threat, eps=threat_to_eps[threat], device=device,
        batch_size=batch_size, preprocessing=dataset_to_transform[dataset], n_examples=n_examples,
        aa_state_path=state_path, seed=seed
    )
    base_acc *= 100
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using BetaReLU with beta={beta:.2f}')
        state_path = Path(state_path_format_str.format(beta=beta))
        new_model = replace_module(copy.deepcopy(model), beta, coeff=coeff)
        _, test_acc = benchmark(
            new_model, dataset=dataset, threat_model=threat, eps=threat_to_eps[threat], device=device,
            batch_size=batch_size, preprocessing=dataset_to_transform[dataset], n_examples=n_examples,
            aa_state_path=state_path, seed=seed
        )
        test_acc *= 100
        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)

    acc_list.append(base_acc)
    beta_list.append(1)

    logger.info(f'Best robust accuracy for {dataset} with {threat} attack: {best_acc:.2f} with beta={best_beta:.2f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_acc_vs_beta(acc_list, beta_list, base_acc, dataset, model_name, f'{threat}_{n_examples}')
