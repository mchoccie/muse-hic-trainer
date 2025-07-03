import os
import cooler
import numpy as np
import pybedtools
from scipy.ndimage import gaussian_filter
lowres_bins   = 256           # 50-kb windows
lowres_res    = 50_000

highres_bins  = 512           # 25-kb windows
highres_res   = 25_000

lowres_maps   = []
highres_maps  = []
coords        = []            # master list of (chrom, start, end)

for fname in os.listdir(data_folder):
    if not fname.endswith(".mcool"):
        continue

    clr_lo = cooler.Cooler(f"{data_folder}/{fname}::resolutions/{lowres_res}")
    clr_hi = cooler.Cooler(f"{data_folder}/{fname}::resolutions/{highres_res}")

    for chrom in clr_lo.chromnames:
        if chrom == "chrY":          # skip chrY if you wish
            continue

        mat_lo = clr_lo.matrix(balance=True).fetch(chrom)
        mat_hi = clr_hi.matrix(balance=True).fetch(chrom)

        # stride must be expressed in **bins**, so make them compatible
        for i in range(0, mat_lo.shape[0] - lowres_bins + 1, stride):   # stride in low-res bins
            start_bp = i * lowres_res
            end_bp   = (i + lowres_bins) * lowres_res

            # (optional) mappability gate
            # if not window_passes(chrom, start_bp, end_bp):
            #     continue

            lo = mat_lo[i:i+lowres_bins, i:i+lowres_bins]
            j  = i * lowres_res // highres_res        # same genomic start in hi-res bins
            hi = mat_hi[j:j+highres_bins, j:j+highres_bins]

            # basic sparsity filter on **each** map
            if (lo == 0).all() or (hi == 0).all():
                continue

            lowres_maps.append(lo.astype(np.float32)[None, ...])
            highres_maps.append(hi.astype(np.float32)[None, ...])
            coords.append((chrom, start_bp, end_bp))
