import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import (MLP, replace_module, ReplacementMapping, get_file_name, fix_seed, logger, set_logger,
                         get_log_file_path)


# Adapted from https://github.com/RandallBalestriero/POLICE
def plot_classification_case(
    width: int, depth: int, training_steps=2000, beta=0.5, ax=None
) -> None:
    N = 1024    # Number of training points
    r = 1

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

    points = points.cuda()

    target = torch.from_numpy(np.array([0] * (N // 2) + [1] * (N // 2))).long().cuda()

    # model and optimizer definition
    model = MLP(2, depth, width, nn.ReLU()).cuda()
    if beta != 1:
        logger.debug(f"Using BetaReLU with beta={beta}")
        replacement_mapping = ReplacementMapping(beta=beta)
        model = replace_module(model, replacement_mapping)
    else:
        logger.debug("Using ReLU")
    optim = torch.optim.AdamW(model.parameters(), 0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=training_steps // 4, gamma=0.3)

    # training
    with tqdm(total=training_steps // 100) as pbar:
        for i in range(training_steps):
            output = model(points)[:, 0]
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
        pred = model(grid).cpu().numpy()
        grid = grid.cpu().numpy()
        points = points.cpu().numpy()
        target = target.cpu().numpy()

    # plot training data
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=["purple" if l == 0 else "orange" for l in target],
        alpha=0.45,
    )

    # plot our decision boundary
    ax.contour(
        xx,
        yy,
        pred[:, 0].reshape((mesh_dim, mesh_dim)),
        levels=[0],
        colors=["red"],
        linewidths=[4],
    )

    # small beautifying process and figure saving
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Beta: {beta:.3f}")


if __name__ == "__main__":
    beta_vals = np.arange(0, 1 + 1e-6, 0.125)
    width = 128
    depth = 2
    training_steps = 2000

    # Create a figure for the 3x3 grid of subplots
    fig, axs = plt.subplots(3, 3, figsize=(30, 30))
    axs = axs.flatten()

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_width{width}_depth{depth}_steps{training_steps}_seed42')

    fix_seed(42)
    for i, beta in enumerate(beta_vals):
        plot_classification_case(width, depth, training_steps, beta, axs[i])

    # Adjust layout and save the combined figure
    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(f'./figures/{get_file_name(get_log_file_path())}.png')
    plt.show()
