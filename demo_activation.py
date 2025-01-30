import torch
import matplotlib.pyplot as plt
from utils.curvature_tuning import BetaAgg


# Create a set of x-values over which to evaluate the activation
x_vals = torch.linspace(-5, 5, steps=200)

# Plot for several beta values
plt.figure(figsize=(7, 5))
betas = [0.5, 0.9]

cmap = plt.colormaps["Dark2"]
colors = [cmap(0), cmap(1), cmap(2)]

for idx, b in enumerate(betas):
    # Instantiate BetaAgg with the current beta
    activation = BetaAgg(beta=b)

    # Forward pass: compute the output for all x_vals
    with torch.no_grad():  # no need for gradients when just plotting
        y_vals = activation(x_vals)

    if b == 0.9:
        # Plot the bold curve (no label)
        plt.plot(x_vals.numpy(), y_vals.numpy(), color=colors[idx], linewidth=5)

        # Plot a duplicate, thinner curve only for the legend
        plt.plot(x_vals.numpy(), y_vals.numpy(), color=colors[idx], linewidth=3, label=r"$\beta$" + f"={b:.1f}")
    else:
        plt.plot(x_vals.numpy(), y_vals.numpy(), color=colors[idx], label=r"$\beta$" + f"={b:.1f}", linewidth=3)

# Also plot ReLU for comparison
relu = torch.nn.ReLU()
with torch.no_grad():
    y_relu = relu(x_vals)
plt.plot(x_vals.numpy(), y_relu.numpy(), color=colors[len(betas)], label=r"$\beta$" + f"=1.00 (ReLU)", linewidth=3)

fontsize = 20
plt.legend(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(True)
plt.savefig("./figures/demo_act.svg", bbox_inches="tight")
plt.show()
