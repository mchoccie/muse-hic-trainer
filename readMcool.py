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



"""
Create *paired* low- and high-resolution Hi-C tiles (256×256 @50 kb
and 512×512 @25 kb) that are perfectly matched base-pair–wise.

Result:
    ├── lowres_dataset.npy      (N, 1, 256, 256)  -- z-scored, float32
    ├── highres_dataset.npy     (N, 1, 512, 512)  -- z-scored, float32
    └── hic_window_coords.npy   (N,) object array  (chrom, start_bp, end_bp)
"""

import os, cooler, numpy as np, pybedtools
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
data_folder       = "Data/"                 # directory with *.mcool files

lowres_res        = 50_000                  # 50-kb resolution
low_bins          = 256                     # 256×256 tiles
low_stride_bins   = 128                     # stride in LOW-RES *bins*

highres_res       = 25_000                  # 25-kb resolution
high_bins         = 512                     # 512×512 tiles
# stride_hi will be computed so windows align

gauss_sigma       = 1.0                     # 0 = disable gaussian blur
min_nonzero_frac  = 0.05                    # ≥5 % non-zero pixels

# ---------- optional mappability filter -------------------------------
mappability_bed   = (
    "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/windows_with_mappability.bed"
)
use_mappability   = False                  # ← flip to True to enable
min_weighted_map  = 0.50                   # keep if ≥50 % mappability

# ----------------------------------------------------------------------
# helper: mappability gate
# ----------------------------------------------------------------------
if use_mappability:
    pass_bed = pybedtools.BedTool(mappability_bed)

    def window_passes(chrom, start, end, thresh=min_weighted_map):
        q = pybedtools.BedTool(f"{chrom}\t{start}\t{end}", from_string=True)
        total_len = end - start
        score = 0.0
        for iv in pass_bed.intersect(q, wao=True):
            ov_len = int(iv[-1])
            try:
                m = float(iv[3])
            except ValueError:
                continue
            score += m * ov_len
        return (score / total_len) >= thresh
else:
    def window_passes(*_args, **_kw):      # always passes
        return True

# ----------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------
low_maps, high_maps, coords = [], [], []
low_stride_bp = low_stride_bins * lowres_res
high_stride_bins = low_stride_bp // highres_res  # ensures alignment

for fname in sorted(os.listdir(data_folder)):
    if not fname.endswith(".mcool"):
        continue

    print(f"\n>>> {fname}")
    clr_lo = cooler.Cooler(f"{data_folder}/{fname}::resolutions/{lowres_res}")
    clr_hi = cooler.Cooler(f"{data_folder}/{fname}::resolutions/{highres_res}")

    for chrom in clr_lo.chromnames:
        if chrom == "chrY":          # skip chrY if desired
            continue

        mat_lo = clr_lo.matrix(balance=True).fetch(chrom)
        mat_hi = clr_hi.matrix(balance=True).fetch(chrom)

        kept = 0
        for i in range(0, mat_lo.shape[0] - low_bins + 1, low_stride_bins):
            # genomic coordinates of the LOW-RES window
            start_bp = i * lowres_res
            end_bp   = start_bp + low_bins * lowres_res

            if not window_passes(chrom, start_bp, end_bp):
                continue

            lo = mat_lo[i : i + low_bins, i : i + low_bins]

            # corresponding index in HIGH-RES bins
            j = (start_bp // highres_res)
            hi = mat_hi[j : j + high_bins, j : j + high_bins]

            # sparsity check on *each* map
            if (np.count_nonzero(lo) / lo.size < min_nonzero_frac or
                np.count_nonzero(hi) / hi.size < min_nonzero_frac):
                continue

            # basic preprocessing ------------------------------------------------
            lo = np.nan_to_num(np.log1p(lo), nan=0.0)
            hi = np.nan_to_num(np.log1p(hi), nan=0.0)

            if gauss_sigma > 0:
                lo = gaussian_filter(lo, sigma=gauss_sigma)
                hi = gaussian_filter(hi, sigma=gauss_sigma)

            # per-window z-score (good for VQGAN / MaskGit training)
            lo = (lo - lo.mean()) / (lo.std() + 1e-6)
            hi = (hi - hi.mean()) / (hi.std() + 1e-6)
            # --------------------------------------------------------------------

            low_maps.append(lo.astype(np.float32)[None, ...])   # shape (1,256,256)
            high_maps.append(hi.astype(np.float32)[None, ...])  # shape (1,512,512)
            coords.append((chrom, start_bp, end_bp))
            kept += 1

        print(f"  {chrom}: kept {kept} windows")

# ----------------------------------------------------------------------
# save
# ----------------------------------------------------------------------
if not low_maps:
    print("\nNo windows satisfied the filters – nothing saved.")
    raise SystemExit

low_arr  = np.stack(low_maps)    # (N,1,256,256)
high_arr = np.stack(high_maps)   # (N,1,512,512)
coords_arr = np.array(coords, dtype=object)

print(f"\nLOW  dataset : {low_arr.shape}")
print(f"HIGH dataset : {high_arr.shape}")
print(f"coords       : {coords_arr.shape}")

np.save("lowres_dataset.npy",  low_arr)
np.save("highres_dataset.npy", high_arr)
np.save("hic_window_coords.npy", coords_arr)

print("\nDone ✔")