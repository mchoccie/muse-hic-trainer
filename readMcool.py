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



import os, cooler, numpy as np, pybedtools
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------------
# 0.  User-tuneable parameters
# ------------------------------------------------------------------------
data_folder   = "/scratch/rnd-rojas/Manan/Data"            # folder with *.mcool
lowres_res    = 50_000             # bp / bin
highres_res   = 25_000
low_bins      = 256                # 256 × 50 kb  → 12.8 Mb window
high_bins     = 512                # 512 × 25 kb  → 12.8 Mb window
low_stride_b  = 128                # stride in *low-res bins*
high_stride_b = low_stride_b * (lowres_res // highres_res)   # 256 bins
min_nonzero_frac = 0.05            # drop if >95 % zeros
gauss_sigma   = 1.0                # set to 0.0 to skip smoothing

# mappability BED (produced with `bedtools map ...`)
mappability_bed = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/windows_with_mappability.bed"
pass_bed  = pybedtools.BedTool(mappability_bed)    # big file? cache on SSD

def window_passes(chrom, start, end, min_weight=0.5):
    """Return True if <chrom:start-end> averages ≥ min_weight mappability."""
    q   = pybedtools.BedTool(f"{chrom}\t{start}\t{end}", from_string=True)
    ovl = pass_bed.intersect(q, wao=True)

    num     = 0.0
    win_len = end - start
    for rec in ovl:
        try:
            mapp = float(rec[3])
        except ValueError:          # “.” means no score
            continue
        num += mapp * int(rec[-1])  # score × overlap length

    return (num / win_len) >= min_weight
# ------------------------------------------------------------------------

low_maps, high_maps, coords = [], [], []

for fname in sorted(os.listdir(data_folder)):
    if not fname.endswith(".mcool"):
        continue

    print(f"\n>>> {fname}")
    clr_lo = cooler.Cooler(f"{data_folder}/{fname}::resolutions/{lowres_res}")
    clr_hi = cooler.Cooler(f"{data_folder}/{fname}::resolutions/{highres_res}")

    for chrom in clr_lo.chromnames:
        if chrom == "chrY":    # skip chrY if desired
            continue
        mat_lo = clr_lo.matrix(balance=True).fetch(chrom)
        mat_hi = clr_hi.matrix(balance=True).fetch(chrom)

        kept = 0
        for i in range(0, mat_lo.shape[0] - low_bins + 1, low_stride_b):
            # map low-res index → genomic range
            start_bp = i * lowres_res
            end_bp   = (i + low_bins) * lowres_res

            # ✅ mappability gate  (uncomment to re-enable)
            # if not window_passes(chrom, start_bp, end_bp): 
            #     continue

            # slice the aligned windows
            lo = mat_lo[i:i+low_bins, i:i+low_bins]
            j  = i * lowres_res // highres_res          # high-res start (bins)
            hi = mat_hi[j:j+high_bins, j:j+high_bins]

            # skip if either map is too sparse
            if (np.count_nonzero(lo) / lo.size < min_nonzero_frac or
                np.count_nonzero(hi) / hi.size < min_nonzero_frac):
                continue

            # basic preprocessing
            lo = np.nan_to_num(np.log1p(lo), nan=0.0)
            hi = np.nan_to_num(np.log1p(hi), nan=0.0)
            if gauss_sigma > 0:
                lo = gaussian_filter(lo, sigma=gauss_sigma)
                hi = gaussian_filter(hi, sigma=gauss_sigma)

            low_maps.append(lo.astype(np.float32)[None, ...])   # [1,256,256]
            high_maps.append(hi.astype(np.float32)[None, ...])  # [1,512,512]
            coords.append((chrom, start_bp, end_bp))
            kept += 1

        print(f"  {chrom}: kept {kept} windows")

# ------------------------------------------------------------------------
# save to .npy
# ------------------------------------------------------------------------
if not low_maps:
    raise RuntimeError("No windows passed the filters!")

low_ds   = np.stack(low_maps)      # (N, 1, 256, 256)
high_ds  = np.stack(high_maps)     # (N, 1, 512, 512)
coords_a = np.array(coords, dtype=object)

print("\nFinal shapes:")
print("  low-res :", low_ds.shape)
print("  high-res:", high_ds.shape)

np.save("hic_dataset_50kb.npy",  low_ds)
np.save("hic_dataset_25kb.npy",  high_ds)
np.save("hic_window_coords.npy", coords_a)