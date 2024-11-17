import os
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from utils.utils import MLP, replace_module, ReplacementMapping, get_file_name, fix_seed
from matplotlib.colors import ListedColormap


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


# Adapted function for plotting classification
def plot_classification_case(
    width: int, depth: int, training_steps=2000, beta_vals=[0.9], noise=0.4, n_turns=3
) -> None:
    N = 1024  # Number of training points

    # data generation
    print("Generating data...")
    X, y = generate_spiral_data(n_points=N, noise=noise, n_turns=n_turns)
    points = torch.from_numpy(X).float().cuda()
    target = torch.from_numpy(y).long().cuda()

    # model and optimizer definition
    relu_model = MLP(2, depth, width, nn.ReLU()).cuda()
    optim = torch.optim.AdamW(relu_model.parameters(), 0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.3)

    # training
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
    num_plots = len(beta_vals) + 1
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    # Plot for ReLU (beta = 1)
    with torch.no_grad():
        pred = relu_model(grid).cpu().numpy()
    axs[0].scatter(
        points.cpu().numpy()[:, 0],
        points.cpu().numpy()[:, 1],
        c=["purple" if l == 0 else "orange" for l in target.cpu().numpy()],
        alpha=0.45,
        edgecolors="none",
    )
    axs[0].contour(
        xx,
        yy,
        pred[:, 0].reshape((mesh_dim, mesh_dim)),
        levels=[0],
        colors=["red"],
        linewidths=[2],
    )
    axs[0].set_title("ReLU (Beta=1)")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Plot for each BetaReLU
    for i, beta in enumerate(beta_vals):
        print(f"Using BetaReLU with beta={beta: .1f}")
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(relu_model)
        new_model = replace_module(orig_model, replacement_mapping)
        with torch.no_grad():
            pred = new_model(grid).cpu().numpy()
        axs[i + 1].scatter(
            points.cpu().numpy()[:, 0],
            points.cpu().numpy()[:, 1],
            c=["purple" if l == 0 else "orange" for l in target.cpu().numpy()],
            alpha=0.45,
            edgecolors="none",
        )
        axs[i + 1].contour(
            xx,
            yy,
            pred[:, 0].reshape((mesh_dim, mesh_dim)),
            levels=[0],
            colors=["blue"],
            linewidths=[2],
        )
        axs[i + 1].set_title(f"BetaReLU (Beta={beta: .2f})")
        axs[i + 1].set_xticks([])
        axs[i + 1].set_yticks([])

    # Adjust layout and save the figure
    plt.tight_layout()
    output_folder = os.path.join("./")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"test.png"))
    plt.show()


if __name__ == "__main__":
    fix_seed(42)

    beta_vals = [0.7]  # Define beta values for BetaReLU
    width = 20
    depth = 2
    training_steps = 2000
    noise = 0.3
    n_turns = 3

    plot_classification_case(width, depth, training_steps, beta_vals, noise, n_turns)
