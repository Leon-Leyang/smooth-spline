import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import MLP, replace_module, ReplacementMapping, get_file_name, fix_seed


# Adapted from https://github.com/RandallBalestriero/POLICE
def plot_classification_case(
    width: int, depth: int, training_steps=2000, beta_vals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> None:
    N = 500    # Number of training points
    r = 1
    train_ratio = 0.5

    # data generation
    print("Generating data...")
    points = []
    for theta in np.linspace(0, 2 * np.pi, N):
        if theta < np.pi:
            x = np.cos(theta) * r - r / 8
            y = np.sin(theta) * r - r / 5
        else:
            x = np.cos(theta) * r + r / 8
            y = np.sin(theta) * r + r / 5
        points.append([x, y])

    points = torch.from_numpy(np.stack(points)).float()
    points += torch.randn_like(points) * 0.025  # Add noise

    points = points.cuda()
    target = torch.from_numpy(np.array([0] * (N // 2) + [1] * (N // 2))).long().cuda()

    # Split into training and testing
    num_train = int(N * train_ratio)
    train_indices = np.random.choice(N, num_train, replace=False)
    test_indices = list(set(range(N)) - set(train_indices))

    train_points = points[train_indices]
    train_target = target[train_indices]
    test_points = points[test_indices]
    test_target = target[test_indices]

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

    # plotting
    domain_bound = 1.8
    mesh_dim = 4000
    with torch.no_grad():
        xx, yy = np.meshgrid(
            np.linspace(-domain_bound, domain_bound, mesh_dim),
            np.linspace(-domain_bound, domain_bound, mesh_dim),
        )
        grid = torch.from_numpy(np.stack([xx.flatten(), yy.flatten()], 1)).float().cuda()
        pred = relu_model(grid).cpu().numpy()

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

    # plot our decision boundary
    plt.contour(
        xx,
        yy,
        pred[:, 0].reshape((mesh_dim, mesh_dim)),
        levels=[0],
        colors=["red"],
        linewidths=[4],
    )

    # small beautifying process and figure saving
    plt.xticks([])
    plt.yticks([])

    # Adjust layout and save the combined figure
    plt.tight_layout()
    output_folder = os.path.join("./")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"test.png"))
    plt.show()


if __name__ == "__main__":
    fix_seed(42)

    beta_vals = np.arange(0, 1, 0.1)
    width = 64
    depth = 2
    training_steps = 2000

    plot_classification_case(width, depth, training_steps, beta_vals)
