import os
import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils import ReplacementMapping, replace_module, get_file_name, test_epoch, get_data_loaders
from resnet import resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def replace_and_test(model, test_loader, beta_vals, mode, dataset):
    """
    Replace ReLU with BetaReLU and test the model on the specified dataset.
    """
    assert mode in ['normal', 'overfit'], 'Mode must be either normal or overfit'

    print('*' * 50)
    print(f'Running post-replace experiment on {dataset}...')
    print('*' * 50)
    criterion = nn.CrossEntropyLoss()

    test_loss_list = []
    beta_list = []

    # Test the original model
    print('Testing the original model...')
    base_test_loss, _ = test_epoch(-1, model, test_loader, criterion, device)
    best_test_loss = base_test_loss
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        print(f'Using BetaReLU with beta={beta:.5f}')
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        test_loss, _ = test_epoch(-1, new_model, test_loader, criterion, device)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_beta = beta
        test_loss_list.append(test_loss)
        beta_list.append(beta)
    test_loss_list.append(base_test_loss)
    beta_list.append(1)
    print(f'Best test loss: {best_test_loss:.6f} with beta={best_beta:.5f}, compared to ReLU test loss: {base_test_loss:.6f}')

    # Plot the test loss vs beta values
    plt.figure(figsize=(12, 6))
    plt.plot(beta_list, test_loss_list)
    plt.axhline(y=base_test_loss, color='r', linestyle='--', label='ReLU Test Loss')
    plt.xlabel('Beta')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Beta Values')
    plt.gca().yaxis.get_major_formatter().set_useOffset(False)
    plt.xticks(beta_list[::5], rotation=45)
    plt.legend()
    output_folder = os.path.join("./figures", get_file_name(__file__))
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"replace_and_test_{dataset}_{mode}.png"))
    plt.show()


def main():
    # Train the model on CIFAR-100
    mode = 'normal'
    dataset = 'cifar100'
    ckpt_folder = os.path.join('./ckpts', mode)
    model = resnet18().to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{dataset}_epoch200.pth')))
    _, test_loader = get_data_loaders(dataset, 128)

    # Replace ReLU with BetaReLU and test the model on the original dataset
    beta_vals = np.arange(0.9995, 1, 0.00001)
    replace_and_test(model, test_loader, beta_vals, mode, dataset)


if __name__ == '__main__':
    main()


