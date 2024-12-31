import torch
import torch.nn as nn
import numpy as np
from utils.eval_post_replace import replace_and_test_acc, replace_and_test_robustness
from utils.data import get_data_loaders
from sklearn.linear_model import LogisticRegression
from utils.utils import get_pretrained_model, get_file_name, fix_seed, result_exists, set_logger, plot_acc_vs_beta
from utils.smooth_spline import replace_module
from train import test_epoch
from loguru import logger
import copy
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeatureExtractor(nn.Module):
    """
    Feature extractor for the model.
    """

    def __init__(self, model, topk=1):
        super().__init__()
        self.base_model = model
        self.topk = topk
        self._features = {}

        self.register_hook(topk)

    def register_hook(self, topk):
        chosen_layers = get_topk_layers(self.base_model, topk)
        for name, module in self.base_model.named_children():
            if module in chosen_layers:
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            self._features[layer_name] = output

        return hook

    def forward(self, x):
        self._features = {}
        self.base_model(x)
        feats_cat = []
        for k, feat in self._features.items():
            feats_cat.append(feat.flatten(1))
        feats_cat = torch.cat(feats_cat, dim=1) if feats_cat else None

        return feats_cat


class WrappedModel(nn.Module):
    def __init__(self, feature_extractor, fc):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.fc = fc

    def forward(self, x):
        feats_cat = self.feature_extractor(x)
        out = self.fc(feats_cat)
        return out


def get_topk_layers(model, topk):
    """
    Get the top-k layers of the model excluding the classification head.
    """
    # Get the top-level children
    children = list(model.children())

    # Exclude the last one if it's the classification head:
    if isinstance(children[-1], nn.Linear):
        children = children[:-1]

    # Now just take the last `topk` from that shortened list
    if topk > len(children):
        topk = len(children)
    return children[-topk:]


def transfer_linear_probe(model, pretrained_ds, transfer_ds, reg=1, topk=1):
    """
    Transfer learning.
    """
    logger.debug('Transferring learning with linear probe...')

    # Get the data loaders
    train_loader, _ = get_data_loaders(f'{pretrained_ds}_to_{transfer_ds}')

    # Remove the last layer of the model
    model = model.to(device)
    feature_extractor = FeatureExtractor(model, topk)
    train_features, train_labels = extract_features(feature_extractor, train_loader)

    # Fit sklearn LogisticRegression as the linear probe
    logistic_regressor = LogisticRegression(max_iter=10000, C=reg)
    logistic_regressor.fit(train_features, train_labels)

    # Replace the last layer of the model with a linear layer
    num_classes = 100 if 'cifar100' in transfer_ds else 1000 if 'imagenet' in transfer_ds else 10

    fc = nn.Linear(logistic_regressor.n_features_in_, num_classes).to(device)
    fc.weight.data = torch.tensor(logistic_regressor.coef_, dtype=torch.float).to(device)
    fc.bias.data = torch.tensor(logistic_regressor.intercept_, dtype=torch.float).to(device)
    fc.weight.requires_grad = False
    fc.bias.requires_grad = False
    model = WrappedModel(feature_extractor, fc)

    logger.debug('Finishing transferring learning...')
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
            features.append(feature.cpu().numpy())
            labels.append(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def lp_then_replace_test_acc(beta_vals, pretrained_ds, transfer_ds, reg=1, coeff=0.5, topk=1):
    """
    Do transfer learning using a linear probe and test the model's accuracy with different beta values of BetaReLU.
    """
    model = get_pretrained_model(pretrained_ds)
    model = transfer_linear_probe(model, pretrained_ds, transfer_ds, reg, topk)
    replace_and_test_acc(model, beta_vals, f'{pretrained_ds}_to_{transfer_ds}', coeff)


def lp_then_replace_test_robustness(threat, beta_vals, pretrained_ds, transfer_ds):
    """
    Do transfer learning using a linear probe and test the model's robustness with different beta values of BetaReLU.
    """
    model = get_pretrained_model(pretrained_ds)
    model = transfer_linear_probe(model, pretrained_ds, transfer_ds, reg=1)
    replace_and_test_robustness(model, threat, beta_vals, f'{pretrained_ds}_to_{transfer_ds}')


def replace_then_lp_test_acc(beta_vals, pretrained_ds, transfer_ds, reg=1, coeff=0.5, topk=1):
    """
    Replace ReLU with BetaReLU and then do transfer learning using a linear probe and test the model's accuracy.
    """
    dataset = f'{pretrained_ds}_to_{transfer_ds}'

    model = get_pretrained_model(pretrained_ds)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = model.__class__.__name__

    _, test_loader = get_data_loaders(dataset)

    logger.info(f'Running replace then linear probe accuracy test for {model_name} on {dataset}...')
    criterion = nn.CrossEntropyLoss()

    acc_list = []
    beta_list = []

    # Test the original model
    logger.debug('Using ReLU...')
    transfer_model = transfer_linear_probe(copy.deepcopy(model), pretrained_ds, transfer_ds, reg, topk)
    _, base_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)
    best_acc = base_acc
    best_beta = 1

    # Test the model with different beta values
    for i, beta in enumerate(beta_vals):
        logger.debug(f'Using BetaReLU with beta={beta:.3f}')
        new_model = replace_module(copy.deepcopy(model), beta, coeff=coeff)
        transfer_model = transfer_linear_probe(new_model, pretrained_ds, transfer_ds, reg, topk)
        _, test_acc = test_epoch(-1, transfer_model, test_loader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_beta = beta
        acc_list.append(test_acc)
        beta_list.append(beta)
    acc_list.append(base_acc)
    beta_list.append(1)
    logger.info(
        f'Best accuracy for {dataset}_replace_lp: {best_acc:.2f} with beta={best_beta:.3f}, compared to ReLU accuracy: {base_acc:.2f}')

    plot_acc_vs_beta(acc_list, beta_list, base_acc, f'{dataset}_replace_lp', model_name)


def test_acc(dataset, beta_vals, coeff):
    """
    Test the model's accuracy with different beta values of BetaReLU on the same dataset.
    """
    model = get_pretrained_model(dataset)
    replace_and_test_acc(model, beta_vals, dataset, coeff)


def test_robustness(threat, dataset, beta_vals):
    """
    Test the model's robustness with different beta values of BetaReLU on the same dataset.
    """
    model = get_pretrained_model(dataset)
    replace_and_test_robustness(model, threat, beta_vals, dataset)


def get_args():
    parser = argparse.ArgumentParser(description='Transfer learning with linear probe')
    parser.add_argument(
        '--order',
        type=str,
        choices=['lp_replace', 'replace_lp'],
        default='lp_replace',
        help='Order of operations: lp_replace or replace_lp'
    )
    parser.add_argument('--coeff', type=float, default=0.5, help='Coefficient for BetaAgg')
    parser.add_argument('--reg', type=float, default=1, help='Regularization strength for Logistic Regression')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--topk', type=int, default=1, help='Number of top-k feature layers to use')
    parser.add_argument('--pretrained_ds', type=str, nargs='+',
                        default=['mnist', 'cifar10', 'cifar100', 'imagenet'], help='List of pretrained datasets')
    parser.add_argument('--transfer_ds', type=str, nargs='+',
                        default=['mnist', 'cifar10', 'cifar100', 'imagenet'], help='List of transfer datasets')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    f_name = get_file_name(__file__)
    log_file_path = set_logger(name=f'{f_name}_{args.order}_coeff{args.coeff}_topk{args.topk}_reg{args.reg}_seed{args.seed}')
    logger.info(f'Log file: {log_file_path}')

    betas = np.arange(0.5, 1 - 1e-6, 0.01)

    pretrained_datasets = args.pretrained_ds
    transfer_datasets = args.transfer_ds

    for pretrained_ds in pretrained_datasets:
        for transfer_ds in transfer_datasets:
            fix_seed(args.seed)  # Fix the seed each time

            if pretrained_ds == transfer_ds:  # Test on the same dataset
                if result_exists(f'{pretrained_ds}'):
                    logger.info(f'Skipping {pretrained_ds} to {transfer_ds} as result already exists.')
                    continue
                else:
                    test_acc(pretrained_ds, betas, args.coeff)
            elif transfer_ds == 'imagenet':  # Skip transfer learning on ImageNet
                continue
            else:  # Test on different datasets
                if args.order == 'lp_replace':
                    if result_exists(f'{pretrained_ds}_to_{transfer_ds}'):
                        logger.info(f'Skipping lp_replace {pretrained_ds} to {transfer_ds} as result already exists.')
                        continue
                    lp_then_replace_test_acc(betas, pretrained_ds, transfer_ds, args.reg, args.coeff, args.topk)
                elif args.order == 'replace_lp':
                    if result_exists(f'{pretrained_ds}_to_{transfer_ds}', replace_then_lp=True):
                        logger.info(f'Skipping replace_lp {pretrained_ds} to {transfer_ds} as result already exists.')
                        continue
                    replace_then_lp_test_acc(betas, pretrained_ds, transfer_ds, args.reg, args.coeff, args.topk)
                else:
                    raise ValueError(f'Invalid order: {args.order}')
