import cooler
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from scipy.ndimage import gaussian_filter

def slice_windows(clr, chromosomes, window_bins=256, stride=128):
    samples = []

    for chrom in chromosomes:
        print(f"  Processing {chrom}")
        mat = clr.matrix(balance=True).fetch(chrom)

        if mat.shape[0] < window_bins:
            print(f"    Skipping {chrom}: only {mat.shape[0]} bins")
            continue

        mat = np.log1p(mat)
        mat = np.nan_to_num(mat, nan=0.0)

        count = 0
        for i in range(0, mat.shape[0] - window_bins + 1, stride):
            w = mat[i:i+window_bins, i:i+window_bins]

            # Optional: Filter out very sparse windows
            if np.count_nonzero(w) / w.size < 0.05:
                continue

            w = (w - np.mean(w)) / (np.std(w) + 1e-6)
            samples.append(w.astype(np.float32)[None, ...])
            count += 1

        print(f"    Kept {count} windows from {chrom}")
    return samples

data_folder = "Data/"
resolution = 50000
window_bins = 256
stride = 128

all_samples = []
for fname in os.listdir(data_folder):
    if fname.endswith(".mcool"):
        fpath = os.path.join(data_folder, fname)
        cooler_path = f"{fpath}::resolutions/{resolution}"
        print(f"Processing file: {fname}")

        try:
            clr = cooler.Cooler(cooler_path)
            chroms = [c for c in clr.chromnames if c != "chrY"]
            samples = slice_windows(clr, chroms, window_bins, stride)
            all_samples.extend(samples)
        except Exception as e:
            print(f"  Failed to process {fname}: {e}")
#sliced_dataset =  slice_windows(clr, chroms, resolution=resolution, window_bins=256, stride=128)
if all_samples:
    dataset = np.stack(all_samples)
    print("Final dataset shape:", dataset.shape)
    np.save("hic_dataset_50kb.npy", dataset)
else:
    print("No valid samples were collected.")