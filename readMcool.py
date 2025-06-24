import cooler
import numpy as np
import os
from cooltools.lib.numutils import observed_over_expected, interp_nan, set_diag
from astropy.convolution import Gaussian2DKernel, convolve


import pybedtools
pass_bed = pybedtools.BedTool("windows_pass.bed")

def window_passes(chrom, start, end):
    # BedTool wants zero-based half-open intervals
    return len(pass_bed.intersect(
        pybedtools.BedTool(f"{chrom}\t{start}\t{end}", from_string=True),
        u=True)) > 0

# === CONFIG ===
data_folder = "Data/"
resolution = 50000
window_bins = 256
stride = 128
reference_genome = "hg19"  # just for annotation purposes
kernel = Gaussian2DKernel(x_stddev=2)      # Akita default σ=2 bins
DIAG_OFF = 2  

# Store results
all_samples = []
window_coordinates = []  # <-- NEW: store genomic coordinates

def akita_preprocess(raw_balanced):
    # 1) clip first diagonals to their median
    clip = np.nanmedian(np.diag(raw_balanced, DIAG_OFF))
    for d in range(-DIAG_OFF+1, DIAG_OFF):
        set_diag(raw_balanced, clip, d)
    raw_balanced = np.clip(raw_balanced, 0, clip)

    # 2) obs/exp & log
    exp = observed_over_expected(raw_balanced, ~np.isnan(raw_balanced))[0]
    oe  = np.log(raw_balanced / exp)
    oe  = interp_nan(oe)
    for d in range(-DIAG_OFF+1, DIAG_OFF):
        set_diag(oe, 0, d)

    # 3) Gaussian smooth
    sm  = convolve(oe, kernel)

    # 4) **optional** crop ends or unwrap upper-tri
    return sm.astype(np.float32)

def slice_windows_with_coords(clr, chromosomes, resolution,
                              window_bins=256, stride=128):
    samples, coords = [], []

    for chrom in chromosomes:
        mat_chr = clr.matrix(balance=True).fetch(chrom)
        for i in range(0, mat_chr.shape[0] - window_bins + 1, stride):
            start_bp = i * resolution
            end_bp   = (i + window_bins) * resolution
            if not window_passes(chrom, start_bp, end_bp):
                continue                          # <- mappability filter

            w = mat_chr[i:i+window_bins, i:i+window_bins]
            if np.count_nonzero(w) / w.size < 0.05:    # Akita’s sparsity check
                continue

            w = akita_preprocess(w)              # Akita cleanup
            samples.append(w[None, ...])         # [1,256,256]  (or vector)
            coords.append((chrom, start_bp, end_bp))
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
