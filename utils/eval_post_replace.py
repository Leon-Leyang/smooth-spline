import copy
import os
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from torch import nn as nn
from utils.robustbench import benchmark
from utils.utils import test_epoch, ReplacementMapping, replace_module, get_file_name, DEFAULT_TRANSFORM
from utils.data import get_data_loaders


def replace_and_test_acc(model, beta_vals, mode, dataset, calling_file):
    """
    Replace ReLU with BetaReLU and test the model on the specified dataset.
    """
    assert mode in ['normal', 'suboptimal', 'overfit'], 'Mode must be either normal, suboptimal or overfit'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = model.__class__.__name__

    _, test_loader = get_data_loaders(dataset)

    print('*' * 50)
    print(f'Running post-replace accuracy test for {model_name}-{mode} on {dataset}...')
    print('*' * 50)
    criterion = nn.CrossEntropyLoss()

    acc_list = []
    beta_list = []

    # Test the original model
    print('Using ReLU...')
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
    output_folder = os.path.join("../figures", get_file_name(calling_file))
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"replace_and_test_acc_{model_name}_{dataset}_{mode}.png"))
    plt.show()


def replace_and_test_robustness(model, threat, beta_vals, mode, dataset, calling_file, batch_size=2000, n_examples=1000,
                                transform_test=DEFAULT_TRANSFORM, model_id=None):
    """
    Replace ReLU with BetaReLU and test the model's robustness on RobustBench.
    """
    assert mode in ['normal', 'suboptimal', 'overfit'], 'Mode must be either normal, suboptimal or overfit'
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

    print('*' * 50)
    print(f'Running post-replace robustness test for {model_name}-{mode} on {dataset} with {threat} attack...')
    print(f'Number of examples: {n_examples}')
    print('*' * 50)

    robust_acc_list = []
    beta_list = []

    state_path_format_str = f"./cache/{model_name}_{dataset}_{mode}_{threat}_{n_examples}_{{beta:.2f}}.json"
    os.makedirs('./cache', exist_ok=True)

    # Test the original model
    if not test_only:
        print('Using ReLU...')
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

    if not test_only:
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
        output_folder = os.path.join("../figures", get_file_name(calling_file))
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(os.path.join(output_folder, f"replace_and_test_robustness_{model_name}_{dataset}_{mode}_{threat}_{n_examples}.png"))
        plt.show()