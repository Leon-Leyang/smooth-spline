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
    Generate a spiral dataset with more turns.

    Args:
    - n_points (int): Total number of points in the dataset.
    - noise (float): Amount of noise to add to the points.
    - n_turns (int): Number of turns in the spiral.

    Returns:
    - X (numpy.ndarray): The coordinates of the points (n_points x 2).
    - y (numpy.ndarray): The class labels for the points (0 or 1).
    """
    n = n_points // 2
    theta = np.linspace(0, n_turns * np.pi, n)
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
    width: int, depth: int, training_steps=2000, beta_vals=[0.9], train_ratio=0.7, noise=0.4, test_noise=0.3, n_turns=3
) -> None:
    N = 1024    # Number of training points

    # data generation
    print("Generating data...")
    X, y = generate_spiral_data(n_points=N, noise=noise, n_turns=n_turns)
    points = torch.from_numpy(X).float()
    target = torch.from_numpy(y).long()

    points = points.cuda()
    target = target.cuda()

    # Split into training and testing
    num_train = int(N * train_ratio)
    train_indices = np.random.choice(N, num_train, replace=False)
    test_indices = list(set(range(N)) - set(train_indices))

    train_points = points[train_indices]
    train_target = target[train_indices]
    test_points = points[test_indices]
    test_target = target[test_indices]

    # Add noise specifically to one class in the test data
    class_indices = test_target == 0
    r = torch.norm(test_points[class_indices], dim=1)  # Calculate radius for each point

    # Apply noise: n times noise for points within n turns
    turn_multipliers = (r / np.pi).clamp(1.0, n_turns)  # Scale noise by the number of turns
    test_points[class_indices] += torch.randn_like(test_points[class_indices]) * (
                test_noise * turn_multipliers[:, None])

    # model and optimizer definition
    relu_model = MLP(2, depth, width, nn.ReLU()).cuda()
    optim = torch.optim.AdamW(relu_model.parameters(), 0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.3)

    # training
    with tqdm(total=training_steps // 100) as pbar:
        for i in range(training_steps):
            output = relu_model(train_points)[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output, train_target.float()
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
        pred = relu_model(grid).cpu().numpy()

    fig, ax = plt.subplots()

    # plot data
    plt.scatter(
        test_points.cpu().numpy()[:, 0],
        test_points.cpu().numpy()[:, 1],
        c=["purple" if l == 0 else "orange" for l in test_target.cpu().numpy()],
        alpha=0.45,
        edgecolors="none",  # No boundary for testing points
    )
    plt.scatter(
        train_points.cpu().numpy()[:, 0],
        train_points.cpu().numpy()[:, 1],
        c=["purple" if l == 0 else "orange" for l in train_target.cpu().numpy()],
        alpha=0.45,
        edgecolors="black",  # Black boundary for training points
        linewidth=0.8,
    )

    # Create a colormap
    cmap = cm.get_cmap("viridis", len(beta_vals))
    num_boundaries = len(beta_vals)
    color_grad = [cmap(i / num_boundaries) for i in range(num_boundaries)]

    # plot our decision boundary
    plt.contour(
        xx,
        yy,
        pred[:, 0].reshape((mesh_dim, mesh_dim)),
        levels=[0],
        colors=["red"],
        linewidths=[2],
    )

    for i, beta in enumerate(beta_vals):
        print(f"Using BetaReLU with beta={beta: .1f}")
        replacement_mapping = ReplacementMapping(beta=beta)
        orig_model = copy.deepcopy(relu_model)
        new_model = replace_module(orig_model, replacement_mapping)
        with torch.no_grad():
            pred = new_model(grid).cpu().numpy()
        plt.contour(
            xx,
            yy,
            pred[:, 0].reshape((mesh_dim, mesh_dim)),
            levels=[0],
            colors=[color_grad[i]],
            linewidths=[2],
        )

    # small beautifying process and figure saving
    plt.xticks([])
    plt.yticks([])

    # Adding a custom color bar
    custom_cmap = ListedColormap(color_grad)
    norm = plt.Normalize(vmin=beta_vals[0], vmax=beta_vals[-1])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks(beta_vals)
    cbar.set_label("Beta values")

    # Adjust layout and save the combined figure
    plt.tight_layout()
    output_folder = os.path.join("./")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"test.png"))
    plt.show()


if __name__ == "__main__":
    fix_seed(42)

    beta_vals = [0.7]
    width = 32
    depth = 2
    training_steps = 2000
    train_ratio = 0.9
    noise = 0.4
    test_noise = 0.6
    n_turns = 3

    plot_classification_case(width, depth, training_steps, beta_vals, train_ratio, noise, test_noise, n_turns)
