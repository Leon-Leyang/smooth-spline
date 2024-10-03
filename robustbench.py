import numpy as np
from robustbench.utils import load_model
from utils import replace_and_test_robustness


def replace_and_test_robustness_on(model_id, threat, beta_vals, dataset):
    n_examples = 10000
    batch_size = 70
    mode = 'normal'
    model = load_model(model_name=model_id, dataset=dataset, threat_model=threat)
    replace_and_test_robustness(model, threat, beta_vals, mode, dataset, __file__, batch_size=batch_size, n_examples=n_examples, transform_test=None)


def main():
    threat = 'Linf'
    beta_vals = np.arange(0.95, 1 - 1e-6, 0.01)
    dataset = 'cifar10'

    for model_id in ['Peng2023Robust', 'Bartoldson2024Adversarial_WRN-94-16']:
        replace_and_test_robustness_on(model_id, threat, beta_vals, dataset)


if __name__ == "__main__":
    main()
