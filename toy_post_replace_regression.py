import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import MLP, replace_module, get_file_name, fix_seed, set_logger, get_log_file_path
from loguru import logger


def generate_curve_data(n_points):
    """
    Generate a Gaussian-like bell curve dataset with noise.
    """
    X = np.linspace(-np.pi / 4, 3 * np.pi / 4, n_points)
    y = np.sin(2 * X)
    return X, y


# Use the helper function in the plotting loop
def plot_classification(
    width: int, depth: int, training_steps=2000, beta_vals=[0.9], c=0.5
) -> None:
    """
    Plot the decision boundary of the model for BetaSwish, BetaSoftplus, and BetaAgg.

    Args:
    - width (int): Width of the MLP.
    - depth (int): Depth of the MLP.
    - training_steps (int): Number of training steps.
    - beta_vals (list of float): List of beta values to test.
    - noise (float): Noise level for spiral data.
    - n_turns (int): Number of spiral turns.
    """
    N = 15  # Number of training points

    # Data generation
    logger.debug("Generating data...")
    X, y = generate_curve_data(N)
    points = torch.from_numpy(X).float().cuda().unsqueeze(-1)
    target = torch.from_numpy(y).float().cuda().unsqueeze(-1)

    # Model and optimizer definition
    relu_model = MLP(1, 1, depth, width, nn.ReLU()).cuda()
    optim = torch.optim.AdamW(relu_model.parameters(), 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.5)

    # Training
    with tqdm(total=training_steps // 100) as pbar:
        for i in range(training_steps):
            output = relu_model(points)
            loss = nn.MSELoss()(output, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            if i % 100 == 0:
                pbar.update(1)
                pbar.set_description(f"Loss {loss.item()}")

    # Create intervals of x for regression plotting
    x_range = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    x_range_torch = torch.from_numpy(x_range).float().cuda()

    # Create subplots
    num_plots = len(beta_vals) + 1  # Include one column for the ReLU baseline
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:  # Ensure axs is iterable even for a single subplot
        axs = [axs]

    # Row configurations for each activation type
    for col, beta in enumerate([None] + beta_vals):  # None for baseline
        if col == 0:
            model = relu_model
            title = "Before Smooth Spline (ReLU)"
        else:
            logger.debug(f"Using BetaAgg with beta={beta:.1f}")
            model = replace_module(copy.deepcopy(relu_model), beta, coeff=c)
            title = f"After Smooth Spline"

        with torch.no_grad():
            predictions = model(x_range_torch).squeeze().cpu().numpy()

        axs[col].scatter(X, y, label="Data", color="blue", alpha=0.7)
        axs[col].plot(x_range, predictions, label="Regression", color="red")
        axs[col].set_title(title)
        axs[col].set_xlabel("X")
        axs[col].set_ylabel("y")
        axs[col].set_aspect('equal', adjustable='box')
        axs[col].legend()

    # Adjust layout and save the figure
    plt.tight_layout(pad=2)
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(f'./figures/{get_file_name(get_log_file_path())}_regression.png')
    plt.show()


if __name__ == "__main__":
    beta_vals = [0.9]
    width = 64
    depth = 4
    training_steps = 10000
    coeff = 0.5

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_width{width}_depth{depth}_steps{training_steps}_coeff{coeff}_seed42')
    fix_seed(42)
    plot_classification(width, depth, training_steps, beta_vals, coeff)
