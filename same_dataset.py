import os
import torch
import numpy as np
from utils import get_data_loaders, replace_and_test
from resnet import resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    mode = 'normal'
    dataset = 'cifar100'
    ckpt_folder = os.path.join('./ckpts', mode)
    model = resnet18().to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{dataset}_epoch200.pth'), weights_only=True))
    _, test_loader = get_data_loaders(dataset, 128)

    # Replace ReLU with BetaReLU and test the model on the original dataset
    beta_vals = np.arange(0.9999, 1, 0.000002)
    replace_and_test(model, test_loader, beta_vals, mode, dataset, __file__)

    mode = 'suboptimal'
    ckpt_folder = os.path.join('./ckpts', mode)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{dataset}_epoch200.pth'), weights_only=True))
    beta_vals = np.arange(0.975, 1, 0.0005)
    replace_and_test(model, test_loader, beta_vals, mode, dataset, __file__)


if __name__ == '__main__':
    main()


