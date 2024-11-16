import argparse
import torch
import numpy as np
from utils.eval_post_replace import replace_and_test_acc, replace_and_test_robustness
from utils.utils import get_pretrained_model, fix_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def replace_and_test_acc_on(mode, dataset, beta_vals):
    model = get_pretrained_model(pretrained_ds=dataset, mode=mode)
    best_beta, best_acc = replace_and_test_acc(model, beta_vals, mode, dataset, __file__)
    return best_beta, best_acc


def replace_and_test_robustness_on(mode, threat, beta_vals, dataset):
    model = get_pretrained_model(pretrained_ds=dataset, mode=mode)
    replace_and_test_robustness(model, threat, beta_vals, mode, dataset, __file__)


def main(args):
    result_file_dir = f'exp/cross_dataset/seed{args.seed}'

    mode_2_beta_vals_acc = {
        'normal': np.arange(0.95, 1 - 1e-6, 0.001),
        'suboptimal': np.arange(0.95, 1 - 1e-6, 0.001),
        'overfit': np.arange(0.95, 1 - 1e-6, 0.001)
    }

    datasets = ['cifar100', 'imagenet', 'cifar10', 'mnist']
    mode = 'normal'
    for ds in datasets:
        fix_seed(args.seed)
        best_beta, best_acc = replace_and_test_acc_on(mode, ds, mode_2_beta_vals_acc[mode])
        with open(f'{result_file_dir}/lp_replace_results.txt', 'a') as f:
            f.write(f'{ds} to {ds}: {best_acc:.2f} with beta={best_beta:.3f}\n')
        with open(f'{result_file_dir}/replace_lp_results.txt', 'a') as f:
            f.write(f'{ds} to {ds}: {best_acc:.2f} with beta={best_beta:.3f}\n')


def get_args():
    parser = argparse.ArgumentParser(description='Same dataset post-replace test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
