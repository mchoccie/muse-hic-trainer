# import cooler, numpy as np, os, pybedtools
# from pathlib import Path

# # ------------------------------------------------------------------
# # mappability BED (already produced with bedtools map/awk ≥0.7)
# pass_bed = pybedtools.BedTool("/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/windows_with_mappability.bed")

# def window_passes(chrom, start, end):
#     """Return True if (chrom,start,end) is in the pass BED."""
#     return bool(pass_bed.intersect(
#         pybedtools.BedTool(f"{chrom}\t{start}\t{end}", from_string=True),
#         u=True))

# # ------------------------------------------------------------------
# # CONFIG
# data_folder   = "Data/"
# resolution    = 50000      # 50-kb bins
# window_bins   = 256
# stride        = 128
# v_min_frac    = 0.05        # keep if ≥5 % non-zero
# # ------------------------------------------------------------------

# all_samples, window_coordinates = [], []

# def slice_windows_with_coords(clr, chromosomes, resolution,
#                               window_bins=256, stride=128):
#     samples, coords = [], []

#     for chrom in chromosomes:
#         mat_chr = clr.matrix(balance=True).fetch(chrom)

#         for i in range(0, mat_chr.shape[0] - window_bins + 1, stride):
#             start_bp = i * resolution
#             end_bp   = (i + window_bins) * resolution

#             # ---- 1. mappability filter
#             if not window_passes(chrom, start_bp, end_bp):
#                 continue

#             w = mat_chr[i:i+window_bins, i:i+window_bins]

#             # ---- 2. skip if entire row/col is NaN   (bad region)
#             if np.any(np.all(np.isnan(w), axis=0)) or np.any(np.all(np.isnan(w), axis=1)):
#                 continue

#             # ---- 3. skip if too sparse (<5 % non-zero)
#             if np.count_nonzero(~np.isnan(w)) / w.size < v_min_frac:
#                 continue

#             # ---- 4. basic log-transform   (no Akita obs/exp etc.)
#             w = np.nan_to_num(w, nan=0.0)   # replace NaNs with 0
#             w = np.log1p(w).astype(np.float32)

#             samples.append(w[None, ...])          # [1, 256, 256]
#             coords.append((chrom, start_bp, end_bp))

#     return samples, coords

# # ------------------------------------------------------------------
# for fname in os.listdir(data_folder):
#     if not fname.endswith(".mcool"):
#         continue

#     clr = cooler.Cooler(f"{data_folder}/{fname}::resolutions/{resolution}")
#     chroms = [c for c in clr.chromnames if c != "chrY"]

#     print(f"Processing {fname}")
#     samples, coords = slice_windows_with_coords(clr, chroms,
#                                                 resolution, window_bins, stride)
#     all_samples.extend(samples)
#     window_coordinates.extend(coords)

# # ------------------------------------------------------------------
# if all_samples:
#     dataset = np.stack(all_samples)  # (N, 1, 256, 256)
#     print("Final dataset shape:", dataset.shape)
#     np.save("hic_dataset_50kb.npy", dataset)
#     np.save("hic_window_coords.npy", np.array(window_coordinates, dtype=object))
# else:
#     print("No valid samples were collected.")



import os
import cooler
import numpy as np
import pybedtools
from scipy.ndimage import gaussian_filter

# === Load mappability BED ===
mappability_bed = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/windows_with_mappability.bed"
pass_bed = pybedtools.BedTool(mappability_bed)

def window_passes(chrom, start, end, min_weighted_mappability=0.5):
    query = pybedtools.BedTool(f"{chrom}\t{start}\t{end}", from_string=True)
    intersected = pass_bed.intersect(query, wao=True)

    total_weighted_overlap = 0
    total_window_length = end - start

    for line in intersected:
        fields = line.fields
        overlap_len = int(fields[-1])
        try:
            mappability = float(fields[3])
        except ValueError:
            continue  # skip if mappability is invalid

        total_weighted_overlap += mappability * overlap_len

    weighted_fraction = total_weighted_overlap / total_window_length
    return weighted_fraction >= min_weighted_mappability

# === Hi-C Slicing Function ===
def slice_windows_with_coords(clr, chromosomes, resolution, window_bins=256, stride=128):
    samples, coords = [], []

    for chrom in chromosomes:
        print(f"  Processing {chrom}")
        mat = clr.matrix(balance=True).fetch(chrom)

        if mat.shape[0] < window_bins:
            print(f"    Skipping {chrom}: only {mat.shape[0]} bins")
            continue

        count = 0
        for i in range(0, mat.shape[0] - window_bins + 1, stride):
            start_bp = i * resolution
            end_bp   = (i + window_bins) * resolution

            # # ✅ Mappability check
            # if not window_passes(chrom, start_bp, end_bp):
            #     continue

            w = mat[i:i+window_bins, i:i+window_bins]

            # ✅ Filter for sparse data
            if np.count_nonzero(w) / w.size < 0.05:
                continue

            w = gaussian_filter(np.nan_to_num(np.log1p(w), nan=0.0), sigma=1.0)
            samples.append(w.astype(np.float32)[None, ...])
            coords.append((chrom, start_bp, end_bp))
            count += 1

        print(f"    Kept {count} windows from {chrom}")
    return samples, coords

# === Main Processing ===
data_folder = "Data/"
resolution = 25000
window_bins = 512
stride = 128

all_samples, window_coordinates = [], []
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

# === Save dataset
if all_samples:
    dataset = np.stack(all_samples)
    print("Final dataset shape:", dataset.shape)
    np.save("hic_dataset_25kb.npy", dataset)
    np.save("hic_window_coords.npy", np.array(window_coordinates, dtype=object))
else:
    print("No valid samples were collected.")