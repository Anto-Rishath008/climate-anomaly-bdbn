# src/visualize_sample.py

import numpy as np
import matplotlib.pyplot as plt
import random

# Load dataset
print("Loading dataset...")
data = np.load("data_raw/era5_preprocessed.npz")
X, y = data["X"], data["y"]

print(f"Dataset loaded: X shape = {X.shape}, y shape = {y.shape}")

# Pick a random sample
idx = random.randint(0, len(X) - 1)
sample_X, sample_y = X[idx], y[idx]

print(f"Visualizing sample #{idx} ...")

# Plot the sequence of 5 timesteps (X) + the target (y)
fig, axes = plt.subplots(1, 6, figsize=(20, 4))

for t in range(5):
    im = axes[t].imshow(sample_X[t], cmap="coolwarm", origin="lower")
    axes[t].set_title(f"Input t-{5 - t}")
    axes[t].axis("off")

im = axes[5].imshow(sample_y, cmap="coolwarm", origin="lower")
axes[5].set_title("Target (t+1)")
axes[5].axis("off")

fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
plt.tight_layout()
plt.show()
