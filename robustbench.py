import numpy as np
from robustbench.utils import load_model
from utils import replace_and_test_robustness


model_id = 'Peng2023Robust'
model = load_model(model_name=model_id, dataset='cifar10', threat_model='Linf')
mode = 'normal'
threat = 'Linf'
beta_vals = np.arange(0.95, 1 - 1e-6, 0.01)
dataset = 'cifar10'
n_examples = 10000
replace_and_test_robustness(
    model, threat, beta_vals, mode, dataset, __file__, batch_size=1000, n_examples=n_examples, transform_test=None
)
