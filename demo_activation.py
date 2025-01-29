import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.curvature_tuning import BetaAgg


# Create a set of x-values over which to evaluate the activation
x_vals = torch.linspace(-5, 5, steps=200)

# Plot for several beta values
plt.figure(figsize=(7, 5))
betas = np.arange(0, 1, 0.25)

cmap = plt.cm.rainbow
colors = [cmap(i) for i in np.linspace(0, 1, len(betas) + 1)]

for idx, b in enumerate(betas):
    # Instantiate BetaAgg with the current beta
    activation = BetaAgg(beta=b)

    # Forward pass: compute the output for all x_vals
    with torch.no_grad():  # no need for gradients when just plotting
        y_vals = activation(x_vals)

    # Plot
    plt.plot(x_vals.numpy(), y_vals.numpy(), color=colors[idx], label=r"$\beta$" + f"={b:.2f}")

# Also plot ReLU for comparison
relu = torch.nn.ReLU()
with torch.no_grad():
    y_relu = relu(x_vals)
plt.plot(x_vals.numpy(), y_relu.numpy(), color=colors[len(betas)], label=r"$\beta$" + f"=1.00 (ReLU)")

plt.title(r"CT with Different $\beta$ Values")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("demo_act.svg", bbox_inches="tight")
plt.show()
