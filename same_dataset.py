import os
import torch
import numpy as np
from utils.eval_post_replace import replace_and_test_acc, replace_and_test_robustness
from utils.utils import get_pretrained_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def replace_and_test_acc_on(mode, dataset, beta_vals):
    model = get_pretrained_model(pretrained_ds=dataset, mode=mode)
    replace_and_test_acc(model, beta_vals, mode, dataset, __file__)


def replace_and_test_robustness_on(mode, threat, beta_vals, dataset):
    model = get_pretrained_model(pretrained_ds=dataset, mode=mode)
    replace_and_test_robustness(model, threat, beta_vals, mode, dataset, __file__)


def main():
    threat_models = ['Linf', 'L2']

    # Replace ReLU with BetaReLU and test the model trained on different conditions on CIFAR-100
    # dataset = 'cifar100'
    mode_2_beta_vals_acc = {
        'normal': np.arange(0.95, 1 - 1e-6, 0.001),
        'suboptimal': np.arange(0.95, 1 - 1e-6, 0.001),
        'overfit': np.arange(0.95, 1 - 1e-6, 0.001)
    }
    mode_2_beta_vals_robustness = {
        'normal': np.arange(0.95, 1 - 1e-6, 0.01),
        'suboptimal': np.arange(0.95, 1 - 1e-6, 0.01),
        'overfit': np.arange(0.95, 1 - 1e-6, 0.01)
    }
    # for mode, beta_vals in mode_2_beta_vals_acc.items():
    #     replace_and_test_acc_on(mode, dataset, beta_vals)
    # for mode, beta_vals in mode_2_beta_vals_robustness.items():
    #     for threat in threat_models:
    #         replace_and_test_robustness_on(mode, threat, beta_vals, dataset)
    #
    # # Replace ReLU with BetaReLU and test the model on different conditions on CIFAR-10
    # dataset = 'cifar10'
    # for mode, beta_vals in mode_2_beta_vals_acc.items():
    #     replace_and_test_acc_on(mode, dataset, beta_vals)
    # for mode, beta_vals in mode_2_beta_vals_robustness.items():
    #     for threat in threat_models:
    #         replace_and_test_robustness_on(mode, threat, beta_vals, dataset)
    #
    # # Replace ReLU with BetaReLU and test the model on different conditions on CIFAR-10
    # dataset = 'noisy_cifar10'
    # replace_and_test_acc_on('normal', dataset, mode_2_beta_vals_acc['normal'])
    # for threat in threat_models:
    #     replace_and_test_robustness_on('normal', threat, mode_2_beta_vals_robustness['normal'], dataset)

    datasets = ['cifar100', 'imagetnet', 'cifar10', 'mnist']
    mode = 'normal'
    for ds in datasets:
        replace_and_test_acc_on(mode, ds, mode_2_beta_vals_acc[mode])


if __name__ == '__main__':
    main()
