import cooler
import numpy as np
import os

# === CONFIG ===
data_folder = "Data/"
resolution = 50000
window_bins = 256
stride = 128
reference_genome = "hg19"  # just for annotation purposes

# Store results
all_samples = []
window_coordinates = []  # <-- NEW: store genomic coordinates

def slice_windows_with_coords(clr, chromosomes, resolution, window_bins=256, stride=128):
    samples = []
    coords = []

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

            if np.count_nonzero(w) / w.size < 0.05:
                continue

            w = (w - np.mean(w)) / (np.std(w) + 1e-6)
            samples.append(w.astype(np.float32)[None, ...])

            # === Calculate genomic coordinates ===
            start = i * resolution
            end = (i + window_bins) * resolution
            coords.append((chrom, start, end))

            count += 1

        print(f"    Kept {count} windows from {chrom}")
    return samples, coords

# === Process each mcool file ===
for fname in os.listdir(data_folder):
    if fname.endswith(".mcool"):
        fpath = os.path.join(data_folder, fname)
        cooler_path = f"{fpath}::resolutions/{resolution}"
        print(f"Processing file: {fname}")

        try:
            clr = cooler.Cooler(cooler_path)
            chroms = [c for c in clr.chromnames if c != "chrY"]
            samples, coords = slice_windows_with_coords(clr, chroms, resolution, window_bins, stride)
            all_samples.extend(samples)
            window_coordinates.extend(coords)
        except Exception as e:
            print(f"  Failed to process {fname}: {e}")

# === Save output ===
if all_samples:
    dataset = np.stack(all_samples)
    print("Final dataset shape:", dataset.shape)

    np.save("hic_dataset_50kb.npy", dataset)
    np.save("hic_window_coords.npy", np.array(window_coordinates, dtype=object))
else:
    print("No valid samples were collected.")
