import copy
import os
from pathlib import Path
import torch
from torch import nn as nn
from utils.robustbench import benchmark
from utils.utils import test_epoch, ReplacementMapping, replace_module, plot_acc_vs_beta, DEFAULT_TRANSFORM, logger
from utils.data import get_data_loaders


def replace_and_test_acc(model, beta_vals, dataset):
    """
    Replace ReLU with BetaReLU and test the model on the specified dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = model.__class__.__name__

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
    logger.info(f'Best accuracy for {dataset}: {best_acc:.2f} with beta={best_beta:.3f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_acc_vs_beta(acc_list, beta_list, base_acc, dataset, model_name)


def replace_and_test_robustness(model, threat, beta_vals, dataset, batch_size=2000, n_examples=1000,
                                transform_test=DEFAULT_TRANSFORM, model_id=None):
    """
    Replace ReLU with BetaReLU and test the model's robustness on RobustBench.
    """
    test_only = len(beta_vals) == 1

    threat_to_eps = {
        'Linf': 8 / 255,
        'L2': 0.5
    }

    model.eval()
    model_name = model_id if model_id is not None else model.__class__.__name__

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if '_' in dataset:
        dataset_to_use = dataset.split('_')[-1]

    logger.info(f'Running post-replace robustness test for {model_name} on {dataset} with {threat} attack...')
    logger.debug(f'Number of examples: {n_examples}')

    acc_list = []
    beta_list = []

    state_path_format_str = f"./cache/{model_name}_{dataset}_{threat}_{n_examples}_{{beta:.2f}}.json"
    os.makedirs('./cache', exist_ok=True)

    # Test the original model
    if not test_only:
        logger.debug('Using ReLU...')
        state_path = Path(state_path_format_str.format(beta=1))
        _, base_acc = benchmark(
            model, dataset=dataset_to_use, threat_model=threat, eps=threat_to_eps[threat], device=device,
            batch_size=batch_size, preprocessing=transform_test, n_examples=n_examples, aa_state_path=state_path
        )
        base_acc *= 100
        best_acc = base_acc
        best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using BetaReLU with beta={beta:.2f}')
        state_path = Path(state_path_format_str.format(beta=beta))
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        _, test_acc = benchmark(
            new_model, dataset=dataset_to_use, threat_model=threat, eps=threat_to_eps[threat], device=device,
            batch_size=batch_size, preprocessing=transform_test, n_examples=n_examples, aa_state_path=state_path
        )
        test_acc *= 100
        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)

    if not test_only:
        acc_list.append(base_acc)
        beta_list.append(1)
        logger.info(f'Best robust accuracy: {best_acc:.2f} with beta={best_beta:.2f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_acc_vs_beta(acc_list, beta_list, base_acc, dataset, model_name, f'{threat}_{n_examples}')
