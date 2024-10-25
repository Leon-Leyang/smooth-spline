import torch
import torch.nn as nn
import numpy as np
from utils.eval_post_replace import replace_and_test_acc, replace_and_test_robustness
from utils.data import get_data_loaders
from sklearn.linear_model import LogisticRegression
from utils.utils import get_pretrained_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transfer_linear_probe(model, mode, pretrained_ds, transfer_ds):
    """
    Transfer learning.
    """
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


def replace_and_test_linear_probe_acc_on(mode, beta_vals, pretrained_ds, transfer_ds):
    """
    Do transfer learning using a linear probe and test the model's accuracy with different beta values of BetaReLU.
    """
    model = get_pretrained_model(pretrained_ds, mode)
    model = transfer_linear_probe(model, mode, pretrained_ds, transfer_ds)
    replace_and_test_acc(model, beta_vals, mode, f'{pretrained_ds}_to_{transfer_ds}', __file__)


def replace_and_test_linear_probe_robustness_on(mode, threat, beta_vals, pretrained_ds, transfer_ds):
    """
    Do transfer learning using a linear probe and test the model's robustness with different beta values of BetaReLU.
    """
    model = get_pretrained_model(pretrained_ds, mode)
    model = transfer_linear_probe(model, mode, f'{pretrained_ds}_to_{transfer_ds}')
    replace_and_test_robustness(model, threat, beta_vals, mode, f'{pretrained_ds}_to_{transfer_ds}', __file__)


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

    pretrained_datasets = ['cifar100', 'imagenet', 'cifar10', 'mnist']
    transfer_datasets = ['cifar100', 'cifar10', 'mnist']
    for pretrained_ds in pretrained_datasets:
        for transfer_ds in transfer_datasets:
            if pretrained_ds == transfer_ds:
                continue
            mode = 'normal'
            replace_and_test_linear_probe_acc_on(mode, mode_2_beta_vals_acc[mode], pretrained_ds, transfer_ds)
            # for threat in threat_models:
            #     replace_and_test_linear_probe_robustness_on(mode, threat, mode_2_beta_vals_robustness[mode], pretrained_ds, transfer_ds)


if __name__ == '__main__':
    main()
