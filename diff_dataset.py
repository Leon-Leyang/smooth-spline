import os
import torch
import copy
import torch.nn as nn
import numpy as np
from resnet import resnet18
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils import get_data_loaders, replace_and_test_acc, ReplacementMapping, replace_module, get_file_name, replace_and_test_robustness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transfer_linear_probe(model, mode, transfer_ds):
    """
    Transfer learning on CIFAR-10 using a linear probe.
    """
    assert mode in ['normal', 'suboptimal', 'overfit'], 'Mode must be either normal, suboptimal or overfit'

    # Get the data loaders for CIFAR-10
    train_loader, test_loader = get_data_loaders('cifar10', train_batch_size=2000)

    # Remove the last layer of the model
    model = model.to(device)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    train_features, train_labels = extract_features(feature_extractor, train_loader)

    # Fit sklearn LogisticRegression as the linear probe
    logistic_regressor = LogisticRegression(max_iter=1000, multi_class='multinomial')
    logistic_regressor.fit(train_features, train_labels)

    # Replace the last layer of the model with a linear layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10).to(device)
    model.fc.weight.data = torch.tensor(logistic_regressor.coef_, dtype=torch.float).to(device)
    model.fc.bias.data = torch.tensor(logistic_regressor.intercept_, dtype=torch.float).to(device)
    model.fc.weight.requires_grad = False
    model.fc.bias.requires_grad = False

    # No further training required
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


def transfer_knn(model):
    """
    Transfer learning on CIFAR-10 using a k-NN classifier.
    """
    # Hyperparameters
    batch_size = 2056
    neighbors = 5

    # Get the data loaders for CIFAR-10
    cifar10_train_loader, cifar10_test_loader = get_data_loaders('cifar10', train_batch_size=batch_size)
    knn = KNeighborsClassifier(n_neighbors=neighbors)

    # Extract features from the pre-trained model
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    train_features, train_labels = extract_features(feature_extractor, cifar10_train_loader)
    test_features, test_labels = extract_features(feature_extractor, cifar10_test_loader)

    # Train the k-NN classifier
    knn.fit(train_features, train_labels)

    # Test the k-NN classifier
    predictions = knn.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions) * 100
    print(f'Accuracy of k-NN classifier: {accuracy:.2f}')
    return accuracy


def get_pretrained_model(pretrained_ds='cifar100', mode='normal'):
    """
    Get the pre-trained model.
    """
    if 'cifar' in pretrained_ds:
        ckpt_folder = os.path.join('./ckpts', mode)
        model = resnet18().to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_{pretrained_ds}_epoch200.pth'), weights_only=True))
    elif pretrained_ds == 'imagenet':
        model = resnet18(weights="IMAGENET1K_V1")

    return model


def replace_and_test_linear_probe_acc_on(mode, beta_vals, pretrained_ds, test_ds):
    """
    Do transfer learning using a linear probe and test the model's accuracy with different beta values of BetaReLU.
    """
    model = get_pretrained_model(pretrained_ds, mode)
    model = transfer_linear_probe(model, mode, test_ds)
    _, test_loader = get_data_loaders(test_ds)
    replace_and_test_acc(model, test_loader, beta_vals, mode, f'{pretrained_ds}_to_{test_ds}', __file__)


def replace_and_test_linear_probe_robustness_on(mode, threat, beta_vals, pretrained_ds, test_ds):
    """
    Do transfer learning using a linear probe and test the model's robustness with different beta values of BetaReLU.
    """
    model = get_pretrained_model(pretrained_ds, mode)
    model = transfer_linear_probe(model, mode)
    replace_and_test_robustness(model, threat, beta_vals, mode, f'{pretrained_ds}_to_{test_ds}', __file__)


def replace_and_test_knn_acc_on(mode, beta_vals):
    """
    Replace ReLU with BetaReLU and test the model's accuracy with different beta values of BetaReLU using a k-NN classifier.
    """
    ckpt_folder = os.path.join('./ckpts', mode)
    model = resnet18().to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_folder, f'resnet18_cifar100_epoch200.pth'), weights_only=True))

    acc_list = []
    beta_list = []

    # Test the original model
    print('Testing the original model')
    base_acc = transfer_knn(model)
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        print(f'Using BetaReLU with beta={beta:.3f}')
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(model)
        new_model = replace_module(orig_model, replacement_mapping)
        acc = transfer_knn(new_model)
        if acc > best_acc:
            best_acc = acc
            best_beta = beta
        acc_list.append(acc)
        beta_list.append(beta)
    acc_list.append(base_acc)
    beta_list.append(1)
    print(f'Best accuracy: {best_acc:.2f} with beta={best_beta:.3f}, compared to ReLU accuracy: {base_acc:.2f}')

    # Plot the test loss vs beta values
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
    output_folder = os.path.join("./figures", get_file_name(__file__))
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"replace_and_test_cifar100_to_cifar10_knn_{mode}.png"))
    plt.show()


def main():
    # Transfer learning on CIFAR-10 using a linear probe and test the model with different beta values of BetaReLU
    threat_models = ['Linf', 'L2']
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
    for mode, beta_vals in mode_2_beta_vals_acc.items():
        replace_and_test_linear_probe_acc_on(mode, beta_vals, pretrained_ds='cifar100', test_ds='cifar10')
    for mode, beta_vals in mode_2_beta_vals_robustness.items():
        for threat in threat_models:
            replace_and_test_linear_probe_robustness_on(mode, threat, beta_vals, pretrained_ds='cifar100', test_ds='cifar10')

    # # Transfer learning on CIFAR-10 using a k-NN classifier and test the model with different beta values of BetaReLU
    # mode_2_beta_vals_acc = {
    #     'normal': np.arange(0.95, 1 - 1e-6, 0.001),
    #     'suboptimal': np.arange(0.95, 1 - 1e-6, 0.001),
    #     'overfit': np.arange(0.95, 1 - 1e-6, 0.001)
    # }
    # for mode, beta_vals in mode_2_beta_vals_acc.items():
    #     replace_and_test_knn_acc_on(mode, beta_vals)


if __name__ == '__main__':
    main()
