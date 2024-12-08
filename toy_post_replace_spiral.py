import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import MLP, replace_module, get_file_name, fix_seed, set_logger, get_log_file_path
from loguru import logger


def generate_spiral_data(n_points, noise=0.5, n_turns=3):
    """
    Generate a spiral dataset with more turns, with reduced noise for smaller radii.

    Args:
    - n_points (int): Total number of points in the dataset.
    - noise (float): Base amount of noise to add to the points.
    - n_turns (int): Number of turns in the spiral.

    Returns:
    - X (numpy.ndarray): The coordinates of the points (n_points x 2).
    - y (numpy.ndarray): The class labels for the points (0 or 1).
    """
    n = n_points // 2
    theta = np.linspace(np.pi / 2, n_turns * np.pi, n)
    r_a = theta
    x_a = r_a * np.cos(theta) + noise * np.random.randn(n)
    y_a = r_a * np.sin(theta) + noise * np.random.randn(n)
    r_b = theta
    x_b = r_b * np.cos(theta + np.pi) + noise * np.random.randn(n)
    y_b = r_b * np.sin(theta + np.pi) + noise * np.random.randn(n)

    X_a = np.vstack((x_a, y_a)).T
    X_b = np.vstack((x_b, y_b)).T
    X = np.vstack((X_a, X_b))
    y = np.hstack((np.zeros(n), np.ones(n)))

    return X, y


def plot_decision_boundary(ax, points, target, xx, yy, pred, title, mesh_dim):
    """
    Plot decision boundary and data points on a single axis.

    Args:
    - ax: The matplotlib axis to plot on.
    - points: Torch tensor of data points.
    - target: Torch tensor of target labels.
    - xx, yy: Meshgrid arrays for plotting the decision boundary.
    - pred: Predictions from the model for the meshgrid.
    - title: Title for the subplot.
    - mesh_dim: Dimension of the meshgrid.
    """
    ax.scatter(
        points.cpu().numpy()[:, 0],
        points.cpu().numpy()[:, 1],
        c=["purple" if l == 0 else "orange" for l in target.cpu().numpy()],
        alpha=0.45,
        edgecolors="none",
    )
    ax.contour(
        xx,
        yy,
        pred[:, 0].reshape((mesh_dim, mesh_dim)),
        levels=[0],
        colors=["red"],
        linewidths=[2],
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# Use the helper function in the plotting loop
def plot_classification_bond(
    width: int, depth: int, training_steps=2000, beta_vals=[0.9], noise=0.4, n_turns=3, c=0.5
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
    N = 1024  # Number of training points

    # Data generation
    logger.debug("Generating data...")
    X, y = generate_spiral_data(n_points=N, noise=noise, n_turns=n_turns)
    points = torch.from_numpy(X).float().cuda()
    target = torch.from_numpy(y).long().cuda()

    # Model and optimizer definition
    relu_model = MLP(2, 2, depth, width, nn.ReLU()).cuda()
    optim = torch.optim.AdamW(relu_model.parameters(), 0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.3)

    # Training
    with tqdm(total=training_steps // 100) as pbar:
        for i in range(training_steps):
            output = relu_model(points)[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output, target.float()
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            if i % 100 == 0:
                pbar.update(1)
                pbar.set_description(f"Loss {loss.item()}")

    domain_bound = np.max(np.abs(X)) * 1.2  # Extend slightly beyond data range
    mesh_dim = 4000
    with torch.no_grad():
        xx, yy = np.meshgrid(
            np.linspace(-domain_bound, domain_bound, mesh_dim),
            np.linspace(-domain_bound, domain_bound, mesh_dim),
        )
        grid = torch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().cuda()

    # Create subplots
    num_plots = len(beta_vals) + 1  # Include one column for the ReLU baseline
    fig, axs = plt.subplots(3, num_plots, figsize=(5 * num_plots, 15))

    # Row configurations for each activation type
    activation_configs = [
        ("BetaSwish", 1),
        ("BetaSoftplus", 0),
        ("BetaAgg", c),
    ]

    for row, (name, coeff) in enumerate(activation_configs):
        # Plot ReLU baseline
        logger.debug("Plotting ReLU baseline")
        with torch.no_grad():
            pred = relu_model(grid).cpu().numpy()
        plot_decision_boundary(
            axs[row, 0], points, target, xx, yy, pred, "ReLU Baseline (Beta=1)", mesh_dim
        )

        # Plot for each beta
        for col, beta in enumerate(beta_vals, start=1):  # Start from second column
            logger.debug(f"Using {name} with beta={beta:.1f}")
            orig_model = copy.deepcopy(relu_model)
            new_model = replace_module(orig_model, beta, coeff=coeff)
            with torch.no_grad():
                pred = new_model(grid).cpu().numpy()
            plot_decision_boundary(
                axs[row, col],
                points,
                target,
                xx,
                yy,
                pred,
                f"{name} Beta={beta:.2f}",
                mesh_dim,
            )

    # Adjust layout and save the figure
    plt.tight_layout(pad=2)
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(f'./figures/{get_file_name(get_log_file_path())}_boundary.png')
    plt.show()


if __name__ == "__main__":
    beta_vals = [0.7, 0.5]  # Define beta values for BetaReLU
    width = 20
    depth = 2
    training_steps = 2000
    noise = 0.3
    n_turns = 3
    coeff = 0.5

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_width{width}_depth{depth}_steps{training_steps}_noise{noise}_turns{n_turns}_coeff{coeff}_seed42')
    fix_seed(42)
    plot_classification_bond(width, depth, training_steps, beta_vals, noise, n_turns, coeff)
