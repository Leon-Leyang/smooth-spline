import os
import torch
import numpy as np
from utils import get_data_loaders, replace_and_test
from resnet import resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def replace_and_test_on(mode, dataset, test_loader, model, beta_vals):
    ckpt_folder = os.path.join('./ckpts', mode)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{dataset}_epoch200.pth'), weights_only=True))
    replace_and_test(model, test_loader, beta_vals, mode, dataset, __file__)


def main():
    model = resnet18().to(device)

    # Replace ReLU with BetaReLU and test the model trained on different conditions on CIFAR-100
    dataset = 'cifar100'
    _, test_loader = get_data_loaders(dataset, 128)
    mode_2_beta_vals = {
        'normal': np.arange(0.9999, 1, 0.000002),
        'suboptimal': np.arange(0.975, 1, 0.0005),
        'overfit': np.arange(0.9999, 1, 0.000002)
    }
    for mode, beta_vals in mode_2_beta_vals.items():
        replace_and_test_on(mode, dataset, test_loader, model, beta_vals)

    # Replace ReLU with BetaReLU and test the model on different conditions on CIFAR-10
    dataset = 'cifar10'
    _, test_loader = get_data_loaders(dataset, 128)
    mode_2_beta_vals = {
        'normal': np.arange(0.9998, 1, 0.000004),
        'suboptimal': np.arange(0.9999, 1, 0.000002),
        'overfit': np.arange(0.9999, 1, 0.000002)
    }
    for mode, beta_vals in mode_2_beta_vals.items():
        replace_and_test_on(mode, dataset, test_loader, model, beta_vals)


if __name__ == '__main__':
    main()
