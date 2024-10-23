import os
import torch
import numpy as np
from utils.eval_post_replace import replace_and_test_acc, replace_and_test_robustness
from utils.data import get_data_loaders
from resnet import resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def replace_and_test_acc_on(mode, dataset, test_loader, model, beta_vals):
    ckpt_folder = os.path.join('./ckpts', mode)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{dataset}_epoch200.pth'), weights_only=True))
    replace_and_test_acc(model, test_loader, beta_vals, mode, dataset, __file__)


def replace_and_test_robustness_on(mode, threat, beta_vals, model, dataset):
    ckpt_folder = os.path.join('./ckpts', mode)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{dataset}_epoch200.pth'), weights_only=True))
    replace_and_test_robustness(model, threat, beta_vals, mode, dataset, __file__)


def main():
    model = resnet18().to(device)
    threat_models = ['Linf', 'L2']

    # Replace ReLU with BetaReLU and test the model trained on different conditions on CIFAR-100
    dataset = 'cifar100'
    _, test_loader = get_data_loaders(dataset, 2056)
    mode_2_beta_vals_acc = {
        'normal': np.arange(0.9, 1 - 1e-6, 0.002),
        'suboptimal': np.arange(0.9, 1 - 1e-6, 0.002),
        'overfit': np.arange(0.9, 1 - 1e-6, 0.002)
    }
    mode_2_beta_vals_robustness = {
        'normal': np.arange(0.95, 1 - 1e-6, 0.01),
        'suboptimal': np.arange(0.95, 1 - 1e-6, 0.01),
        'overfit': np.arange(0.95, 1 - 1e-6, 0.01)
    }
    for mode, beta_vals in mode_2_beta_vals_acc.items():
        replace_and_test_acc_on(mode, dataset, test_loader, model, beta_vals)
    for mode, beta_vals in mode_2_beta_vals_robustness.items():
        for threat in threat_models:
            replace_and_test_robustness_on(mode, threat, beta_vals, model, dataset)

    # Replace ReLU with BetaReLU and test the model on different conditions on CIFAR-10
    dataset = 'cifar10'
    _, test_loader = get_data_loaders(dataset, 2056)
    for mode, beta_vals in mode_2_beta_vals_acc.items():
        replace_and_test_acc_on(mode, dataset, test_loader, model, beta_vals)
    for mode, beta_vals in mode_2_beta_vals_robustness.items():
        for threat in threat_models:
            replace_and_test_robustness_on(mode, threat, beta_vals, model, dataset)

    # Replace ReLU with BetaReLU and test the model on different conditions on CIFAR-10
    dataset = 'noisy_cifar10'
    _, test_loader = get_data_loaders(dataset, 2056)
    replace_and_test_acc_on('normal', dataset, test_loader, model, mode_2_beta_vals_acc['normal'])
    for threat in threat_models:
        replace_and_test_robustness_on('normal', threat, mode_2_beta_vals_robustness['normal'], model, dataset)


if __name__ == '__main__':
    main()
