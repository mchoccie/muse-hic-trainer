import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load your dataset
dataset = np.load("hic_dataset_50kb.npy")
index = 0  # pick which Hi-C window to visualize
hic_map = dataset[index, 0]

# Apply Gaussian blur (tweak sigma for smoothness)
smoothed = gaussian_filter(hic_map, sigma=1.2)

# Plot without gridlines, matching your uploaded image
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(smoothed, cmap='Reds', aspect='equal', vmin=0, vmax=5)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Log-scaled contact intensity')

# Titles and ticks styled to match
ax.set_title("Hi-C Contact Map (log-scaled)", fontsize=12)
ax.tick_params(axis='both', which='both', length=0)  # no tick marks

# Clean layout and save
plt.tight_layout()
plt.savefig("hic_contact_map_matched.png", dpi=300)
plt.close()