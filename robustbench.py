import argparse
import numpy as np
from robustbench.utils import load_model
from utils import replace_and_test_robustness


def replace_and_test_robustness_on(model_id, threat, beta_vals, dataset):
    n_examples = 10000
    batch_size = 70
    mode = 'normal'
    model = load_model(model_name=model_id, dataset=dataset, threat_model=threat)
    replace_and_test_robustness(model, threat, beta_vals, mode, dataset, __file__, batch_size=batch_size, n_examples=n_examples, transform_test=None, model_id=model_id)


def main():
    parser = argparse.ArgumentParser(description="Run robustness tests on models.")
    parser.add_argument('--model_ids', type=str, nargs='+', default=['Peng2023Robust', 'Wang2023Better_WRN-70-16'], help='List of model IDs to test')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use')
    args = parser.parse_args()

    threat = 'Linf'
    beta_vals = np.arange(0.95, 1 - 1e-6, 0.01)
    dataset = args.dataset

    for model_id in args.model_ids:
        replace_and_test_robustness_on(model_id, threat, beta_vals, dataset)


if __name__ == "__main__":
    main()
