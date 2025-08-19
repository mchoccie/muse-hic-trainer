#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build paired Hi-C super-resolution datasets with:
  - High-res tiles: 512x512 @ 25 kb (ground truth)
  - Low-res tiles : 256x256 @ 50 kb (synthetic, via 2x coarsening)
  - Stratum-wise z-score normalization
  - Validity masks (hi & low) and distance maps saved separately

Outputs:
  lowres_dataset.npy        -> (N, 1, 256, 256)  [lo_norm]
  lowres_mask.npy           -> (N, 1, 256, 256)  [lo_mask]
  lowres_distance.npy       -> (N, 1, 256, 256)  [dist_lo]
  highres_dataset.npy       -> (N, 1, 512, 512)  [hi_norm]
  highres_mask.npy          -> (N, 1, 512, 512)  [hi_mask]
  hic_window_coords.npy     -> (N,) object array of (chrom, start_bp, end_bp)
  stratum_stats_hi_25k_512.npz, stratum_stats_lo_50k_256.npz
"""

import os
import numpy as np
import cooler

# --------------------------- Configuration ------------------------------------
data_folder        = "Data"
highres_res        = 25_000
high_bins          = 512

lowres_res         = 50_000
low_bins           = 256
low_stride_bins    = 128

min_valid_frac     = 0.70
skip_chrY          = True

use_mappability    = False
mappability_bed    = "/path/to/windows_with_mappability.bed"
min_weighted_map   = 0.50

save_stratum_stats = True

# ---------------------- Optional mappability gate -----------------------------
def make_window_passes():
    if not use_mappability:
        return lambda *args, **kwargs: True
    try:
        import pybedtools
    except ImportError as e:
        raise ImportError("use_mappability=True requires 'pybedtools'") from e
    pass_bed = pybedtools.BedTool(mappability_bed)
    def window_passes(chrom: str, start: int, end: int, thresh=min_weighted_map):
        q = pybedtools.BedTool(f"{chrom}\t{start}\t{end}", from_string=True)
        total_len = max(1, end - start)
        score = 0.0
        for iv in pass_bed.intersect(q, wao=True):
            ov_len = int(iv[-1])
            try:
                m = float(iv[3])
            except Exception:
                continue
            score += m * ov_len
        return (score / total_len) >= thresh
    return window_passes

window_passes = make_window_passes()

# --------------------------- Helpers ------------------------------------------
def coarsen2_batch(x3: np.ndarray) -> np.ndarray:
    N, H, W = x3.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    x3 = x3[:, :H2, :W2]
    return x3.reshape(N, H2 // 2, 2, W2 // 2, 2).sum(axis=(2, 4))

def maxpool2_mask_batch(m3: np.ndarray) -> np.ndarray:
    N, H, W = m3.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    m3 = m3[:, :H2, :W2]
    return m3.reshape(N, H2 // 2, 2, W2 // 2, 2).max(axis=(2, 4)).astype(np.float32)

def distance_map(n: int) -> np.ndarray:
    i = np.arange(n)
    d = np.abs(i[:, None] - i[None, :]).astype(np.float32)
    return d / max(1.0, d.max())

def fit_stratum_stats(X: np.ndarray, M: np.ndarray):
    N, K, _ = X.shape
    sums = np.zeros(K, dtype=np.float64)
    sums2 = np.zeros(K, dtype=np.float64)
    cnt = np.zeros(K, dtype=np.int64)
    for n in range(N):
        x = np.log1p(X[n])
        m = M[n].astype(bool)
        for d in range(K):
            if K - d <= 0:
                continue
            vals = np.diag(x, k=d)
            mv   = np.diag(m, k=d)
            vals = vals[mv]
            if vals.size == 0:
                continue
            sums[d]  += vals.sum()
            sums2[d] += (vals ** 2).sum()
            cnt[d]   += vals.size
    means = sums / np.maximum(cnt, 1)
    var   = np.maximum(sums2 / np.maximum(cnt, 1) - means ** 2, 1e-6)
    stds  = np.sqrt(var)
    return means.astype(np.float32), stds.astype(np.float32)

def apply_stratum_zscore_batch(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    N, K, _ = X.shape
    idx = np.arange(K)
    D = np.abs(idx[:, None] - idx[None, :])
    Xl = np.log1p(X)
    out = (Xl - means[D]) / (stds[D] + 1e-6)
    return out.astype(np.float32)

# ----------------------------- Extraction -------------------------------------
def main():
    low_stride_bp = low_stride_bins * lowres_res
    high_stride_bins = low_stride_bp // highres_res

    hi_tiles, hi_masks, coords = [], [], []
    mcools = [f for f in sorted(os.listdir(data_folder)) if f.endswith(".mcool")]
    if not mcools:
        raise FileNotFoundError(f"No .mcool files in {data_folder}")

    for fname in mcools:
        print(f"\n>>> {fname}")
        clr_hi = cooler.Cooler(os.path.join(data_folder, f"{fname}::resolutions/{highres_res}"))
        for chrom in clr_hi.chromnames:
            if skip_chrY and chrom == "chrY":
                continue
            mat_hi = clr_hi.matrix(balance=False).fetch(chrom).astype(np.float32)
            kept = 0
            for j in range(0, mat_hi.shape[0] - high_bins + 1, high_stride_bins):
                start_bp = j * highres_res
                end_bp   = start_bp + high_bins * highres_res
                if not window_passes(chrom, start_bp, end_bp):
                    continue
                hi = mat_hi[j:j + high_bins, j:j + high_bins]
                mask = np.isfinite(hi).astype(np.float32)
                if mask.mean() < min_valid_frac:
                    continue
                hi = np.nan_to_num(hi, nan=0.0)
                hi_tiles.append(hi)
                hi_masks.append(mask)
                coords.append((chrom, int(start_bp), int(end_bp)))
                kept += 1
            print(f"  {chrom}: kept {kept} windows")

    if not hi_tiles:
        print("\nNo windows satisfied the filters – nothing saved.")
        return

    hi_tiles = np.stack(hi_tiles)  # (N, 512, 512)
    hi_masks = np.stack(hi_masks)  # (N, 512, 512)
    N = hi_tiles.shape[0]

    print("\n--- After stacking raw tiles ---")
    print("hi_tiles:", hi_tiles.shape, hi_tiles.dtype)
    print("hi_masks:", hi_masks.shape, hi_masks.dtype)
    print("valid frac (mean mask):", float(hi_masks.mean()))

    # Synthetic low-res (2x)
    lo_counts = coarsen2_batch(hi_tiles)      # (N, 256, 256)
    lo_masks  = maxpool2_mask_batch(hi_masks) # (N, 256, 256)

    print("\n--- After coarsening (2x) ---")
    print("lo_counts:", lo_counts.shape, lo_counts.dtype)
    print("lo_masks :", lo_masks.shape,  lo_masks.dtype)

    # Stratum stats (fit on ALL here; in practice fit on TRAIN only)
    hi_means, hi_stds = fit_stratum_stats(hi_tiles, hi_masks)
    lo_means, lo_stds = fit_stratum_stats(lo_counts, lo_masks)

    print("\n--- Stratum stats ---")
    print("hi_means/stds:", hi_means.shape, hi_stds.shape)
    print("lo_means/stds:", lo_means.shape, lo_stds.shape)

    # Apply stratum z-score
    hi_norm = apply_stratum_zscore_batch(hi_tiles, hi_means, hi_stds)  # (N,512,512)
    lo_norm = apply_stratum_zscore_batch(lo_counts, lo_means, lo_stds) # (N,256,256)

    print("\n--- After stratum z-score ---")
    print("hi_norm:", hi_norm.shape, hi_norm.dtype)
    print("lo_norm:", lo_norm.shape, lo_norm.dtype)

    # Package tensors (ONE CHANNEL for images)
    dist_lo = np.broadcast_to(distance_map(low_bins),  (N, low_bins,  low_bins)).astype(np.float32)

    lowres_img   = lo_norm[:, None, :, :].astype(np.float32)     # (N,1,256,256)
    lowres_mask  = lo_masks[:, None, :, :].astype(np.float32)    # (N,1,256,256)
    lowres_dist  = dist_lo[:, None, :, :].astype(np.float32)     # (N,1,256,256)

    highres_img   = hi_norm[:, None, :, :].astype(np.float32)    # (N,1,512,512)
    highres_mask  = hi_masks[:, None, :, :].astype(np.float32)   # (N,1,512,512)

    # Save
    np.save("lowres_dataset_gpt5.npy",        lowres_img)
    np.save("lowres_mask_gpt5.npy",           lowres_mask)
    np.save("lowres_distance_gpt5.npy",       lowres_dist)
    np.save("highres_dataset_gpt5.npy",       highres_img)
    np.save("highres_mask_gpt5.npy",          highres_mask)
    np.save("hic_window_coords_gpt5.npy",     np.array(coords, dtype=object))

    if save_stratum_stats:
        np.savez("stratum_stats_hi_25k_512_gpt5.npz", means=hi_means, stds=hi_stds)
        np.savez("stratum_stats_lo_50k_256_gpt5.npz", means=lo_means, stds=lo_stds)

    # Final prints
    print("\n--- Final tensors ---")
    print("lowres_dataset :", lowres_img.shape,  lowres_img.dtype)
    print("lowres_mask    :", lowres_mask.shape, lowres_mask.dtype)
    print("lowres_distance:", lowres_dist.shape, lowres_dist.dtype)
    print("highres_dataset:", highres_img.shape, highres_img.dtype)
    print("highres_mask   :", highres_mask.shape, highres_mask.dtype)
    print(f"\nSaved {N} datapoints")
    assert N == highres_img.shape[0] == highres_mask.shape[0] == len(coords)

    total_bytes = (lowres_img.nbytes + lowres_mask.nbytes + lowres_dist.nbytes +
                   highres_img.nbytes + highres_mask.nbytes)
    print(f"Estimated RAM for saved arrays: {total_bytes/1e6:.1f} MB")
    print("\nDone ✔")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
