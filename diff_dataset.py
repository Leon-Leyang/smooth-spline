import torch
import torch.nn as nn
import numpy as np
from utils.eval_post_replace import replace_and_test_acc, replace_and_test_robustness
from utils.data import get_data_loaders
from sklearn.linear_model import LogisticRegression
from utils.utils import get_pretrained_model, test_epoch, ReplacementMapping, replace_module, get_file_name, fix_seed
import copy
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transfer_linear_probe(model, mode, pretrained_ds, transfer_ds):
    """
    Transfer learning.
    """
    print('Transferring learning with linear probe...')
    assert mode in ['normal', 'suboptimal', 'overfit'], 'Mode must be either normal, suboptimal or overfit'

    # Get the data loaders
    train_loader, _ = get_data_loaders(f'{pretrained_ds}_to_{transfer_ds}')

    # Remove the last layer of the model
    model = model.to(device)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    train_features, train_labels = extract_features(feature_extractor, train_loader)

    # Fit sklearn LogisticRegression as the linear probe
    logistic_regressor = LogisticRegression(max_iter=10000, multi_class='multinomial')
    logistic_regressor.fit(train_features, train_labels)

    # Replace the last layer of the model with a linear layer
    num_classes = 100 if 'cifar100' in transfer_ds else 1000 if 'imagenet' in transfer_ds else 10
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
    model.fc.weight.data = torch.tensor(logistic_regressor.coef_, dtype=torch.float).to(device)
    model.fc.bias.data = torch.tensor(logistic_regressor.intercept_, dtype=torch.float).to(device)
    model.fc.weight.requires_grad = False
    model.fc.bias.requires_grad = False

    print('Finishing transferring learning...')
    return model


def extract_features(feature_extractor, dataloader):
    """
    Extract features from the model.
    """
    feature_extractor.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feature = feature_extractor(inputs)
            feature = torch.flatten(feature, 1)
            features.append(feature.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def lp_then_replace_test_acc(mode, beta_vals, pretrained_ds, transfer_ds):
    """
    Do transfer learning using a linear probe and test the model's accuracy with different beta values of BetaReLU.
    """
    model = get_pretrained_model(pretrained_ds, mode)
    model = transfer_linear_probe(model, mode, pretrained_ds, transfer_ds)
    best_beta, best_acc = replace_and_test_acc(model, beta_vals, mode, f'{pretrained_ds}_to_{transfer_ds}', __file__)
    return best_beta, best_acc


def replace_and_test_linear_probe_robustness_on(mode, threat, beta_vals, pretrained_ds, transfer_ds):
    """
    Do transfer learning using a linear probe and test the model's robustness with different beta values of BetaReLU.
    """
    model = get_pretrained_model(pretrained_ds, mode)
    model = transfer_linear_probe(model, mode, pretrained_ds, transfer_ds)
    replace_and_test_robustness(model, threat, beta_vals, mode, f'{pretrained_ds}_to_{transfer_ds}', __file__)


def replace_then_lp_test_acc(mode, beta_vals, pretrained_ds, transfer_ds):
    """
    Replace ReLU with BetaReLU and then do transfer learning using a linear probe and test the model's accuracy.
    """
    dataset = f'{pretrained_ds}_to_{transfer_ds}'

    model = get_pretrained_model(pretrained_ds, mode)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = model.__class__.__name__

    _, test_loader = get_data_loaders(dataset)

    print('*' * 50)
    print(f'Running replace then linear probe accuracy test for {model_name}-{mode} on {dataset}...')
    print('*' * 50)
    criterion = nn.CrossEntropyLoss()

    acc_list = []
    beta_list = []

    # Test the original model
    print('Using ReLU...')
    orig_model = copy.deepcopy(model)
    transfer_model = transfer_linear_probe(orig_model, mode, pretrained_ds, transfer_ds)
    _, base_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        print(f'Using BetaReLU with beta={beta:.3f}')
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        transfer_model = transfer_linear_probe(new_model, mode, pretrained_ds, transfer_ds)
        _, test_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)
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
    output_folder = os.path.join("../figures", get_file_name(__file__))
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"replace_and_lp_test_acc_{model_name}_{dataset}_{mode}.png"))
    plt.show()


def main(args):
    result_file_dir = f'exp/cross_dataset/seed{args.manual_seed}'

    mode_2_beta_vals_acc = {
        'normal': np.arange(0.95, 1 - 1e-6, 0.001),
        'suboptimal': np.arange(0.95, 1 - 1e-6, 0.001),
        'overfit': np.arange(0.95, 1 - 1e-6, 0.001)
    }

    pretrained_datasets = ['mnist', 'cifar10', 'cifar100', 'imagenet']
    transfer_datasets = ['mnist', 'cifar10', 'cifar100']
    for pretrained_ds in pretrained_datasets:
        for transfer_ds in transfer_datasets:
            if pretrained_ds == transfer_ds:
                continue
            mode = 'normal'
            fix_seed(args.seed)
            if args.order == 'lp_replace':
                best_beta, best_acc = lp_then_replace_test_acc(mode, mode_2_beta_vals_acc[mode], pretrained_ds, transfer_ds)
                with open(f'{result_file_dir}/lp_replace_results.txt', 'a') as f:
                    f.write(f'{pretrained_ds} to {transfer_ds}: {best_acc:.2f} with beta={best_beta:.3f}\n')
            elif args.order == 'replace_lp':
                best_beta, best_acc = replace_then_lp_test_acc(mode, mode_2_beta_vals_acc[mode], pretrained_ds, transfer_ds)
                with open(f'{result_file_dir}/replace_lp_results.txt', 'a') as f:
                    f.write(f'{pretrained_ds} to {transfer_ds}: {best_acc:.2f} with beta={best_beta:.3f}\n')
            else:
                raise ValueError(f'Invalid order: {args.order}')


def get_args():
    parser = argparse.ArgumentParser(description='Transfer learning with linear probe')
    parser.add_argument(
        '--order',
        type=str,
        choices=['lp_replace', 'replace_lp'],
        default='lp_replace',
        help='Order of operations: lp_replace or replace_lp'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
