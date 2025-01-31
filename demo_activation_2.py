import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import torch

from utils.curvature_tuning import BetaAgg


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Return a new colormap that is a truncated version of `cmap`."""
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", new_colors, N=n
    )


# Truncate the reversed magma colormap so that its "start" is at 0.2
original_cmap = plt.get_cmap("magma_r")  # The reversed magma
trunc_cmap = truncate_colormap(original_cmap, minval=0.2, maxval=1.0)

# Normalize from 0 to 1 in the data/label space
norm = mcolors.Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(norm=norm, cmap=trunc_cmap)
sm.set_array([])

# Create a set of x-values
x_vals = torch.linspace(-3, 3, steps=200)

# Example beta values
betas = np.arange(0, 1.0, 0.1)

# Choose N = len(betas) + 1 colors so that the last color is for ReLU
color_values = np.linspace(0, 1, len(betas) + 1)
colors = [trunc_cmap(cv) for cv in color_values]

# Prepare figure with 3 subplots in a row
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

# Titles for the three cases
titles = [r"Smoothing the Region Assignment", r"Smoothing the Max", r"Combined"]

# Different coefficient settings for each subplot
coeff_values = [1, 0, None]

for ax, title, coeff in zip(axes, titles, coeff_values):
    for idx, b in enumerate(betas):
        activation = BetaAgg(beta=b) if coeff is None else BetaAgg(beta=b, coeff=coeff)
        with torch.no_grad():
            y_vals = activation(x_vals)
        ax.plot(x_vals.numpy(), y_vals.numpy(), color=colors[idx], linewidth=1)

    # Also plot ReLU with the last color
    relu = torch.nn.ReLU()
    with torch.no_grad():
        y_relu = relu(x_vals)
    ax.plot(x_vals.numpy(), y_relu.numpy(), color=colors[-1], linewidth=1)

    # Format axes
    fontsize = 16
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-1, 0, 1, 2, 3])
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.grid(True)
    ax.set_title(title, fontsize=fontsize + 2)

# Adjust layout and add colorbar
fig.subplots_adjust(right=0.85)  # Make room for colorbar
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Position of colorbar
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label(r"$\beta$ values", fontsize=16)
cbar.set_ticks([0, 1])  # Ticks at 0 and 1
cbar.set_ticklabels(["0", "1"])  # Corresponding labels
cbar.ax.tick_params(labelsize=16)

plt.savefig("./figures/Activation Functions.pdf", bbox_inches="tight")
plt.show()
