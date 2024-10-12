import argparse
import numpy as np
from robustbench.utils import load_model
from utils import replace_and_test_robustness


def replace_and_test_robustness_on(model_id, threat, beta_vals, dataset, n_examples, batch_size):
    mode = 'normal'
    model = load_model(model_name=model_id, dataset=dataset, threat_model=threat)
    replace_and_test_robustness(model, threat, beta_vals, mode, dataset, __file__, batch_size=batch_size, n_examples=n_examples, transform_test=None, model_id=model_id)


def main():
    parser = argparse.ArgumentParser(description="Run robustness tests on models.")
    parser.add_argument('--model_ids', type=str, nargs='+', default=['Peng2023Robust', 'Wang2023Better_WRN-70-16'], help='List of model IDs to test')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use')
    parser.add_argument('--n_examples', type=int, default=10000, help='Number of examples to use')
    parser.add_argument('--batch_size', type=int, default=70, help='Batch size for testing')
    parser.add_argument('--beta_range', type=float, nargs=2, default=[0.95, 1], help='Range of beta values to test')
    args = parser.parse_args()

    threat = 'Linf'
    beta_vals = np.arange(args.beta_range[0], args.beta_range[1] - 1e-6, 0.01)
    dataset = args.dataset
    n_examples = args.n_examples
    batch_size = args.batch_size

    for model_id in args.model_ids:
        replace_and_test_robustness_on(model_id, threat, beta_vals, dataset, n_examples, batch_size)


if __name__ == "__main__":
    main()
