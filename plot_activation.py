import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.curvature_tuning import CT

# Initialize variables for the plot
x = torch.linspace(-5, 5, 500)  # Input range
betas = np.arange(0, 1, 0.1)  # Beta values
coeff = 0.5  # Fixed coefficient

# Create the plot
plt.figure(figsize=(10, 6))

for beta in betas:
    activation = CT(beta=beta, coeff=coeff, trainable=False)
    y = activation(x).detach().numpy()
    plt.plot(x.numpy(), y, label=f'beta={beta:.2f}')

# Configure the plot
plt.title('BetaAgg Activation Function for Different Beta Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title='Beta')
plt.grid(True)
plt.show()
