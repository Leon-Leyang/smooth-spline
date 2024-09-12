import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import Network, replace_module, ReplacementMapping
import matplotlib.cm as cm
import copy
from matplotlib.colors import ListedColormap


# Adapted from https://github.com/RandallBalestriero/POLICE
def plot_classification_case(
    width: int, depth: int, training_steps=2000, beta_vals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> None:
    N = 1024    # Number of training points
    r = 1

    # Fix the random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    if not os.path.exists("./data/toy_data.pt"):
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
        points += torch.randn_like(points) * 0.025

        # save the data
        os.makedirs("./data", exist_ok=True)
        torch.save(points, f"./data/toy_data.pt")

        points = points.cuda()
    else:
        print("Using cached data...")
        points = torch.load("./data/toy_data.pt", weights_only=True).cuda()
    target = torch.from_numpy(np.array([0] * (N // 2) + [1] * (N // 2))).long().cuda()

    # model and optimizer definition
    relu_model = Network(2, depth, width, nn.ReLU()).cuda()
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
        points = points.cpu().numpy()
        target = target.cpu().numpy()

    # plot training data
    fig, ax = plt.subplots()
    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=["purple" if l == 0 else "orange" for l in target],
        alpha=0.45,
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
        linewidths=[4],
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
    os.makedirs("./figures", exist_ok=True)
    plt.savefig(f"./figures/post_replace_width{width}_depth{depth}_steps{training_steps}.png")
    plt.show()


if __name__ == "__main__":
    beta_vals = np.arange(0, 1, 0.1)
    width = 128
    depth = 2
    training_steps = 2000

    plot_classification_case(width, depth, training_steps, beta_vals)
