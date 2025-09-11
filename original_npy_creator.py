#!/usr/bin/env python3
# Build a single Hi-C dataset by scanning one or more folders for .mcool/.cool files.
# ⚠️ Normalization logic is UNCHANGED from your snippet (log1p → NaN→0 → z-score).

import os
import argparse
import numpy as np
import cooler


# ------------------------------- core (unchanged normalization) -------------------------------

def slice_windows_with_coords(clr, chromosomes, resolution, window_bins=256, stride=128):
    samples = []
    coords = []

    for chrom in chromosomes:
        print(f"  Processing {chrom}")
        mat = clr.matrix(balance=True).fetch(chrom)

        if mat.shape[0] < window_bins:
            print(f"    Skipping {chrom}: only {mat.shape[0]} bins")
            continue

        # --- normalization (unchanged) ---
        mat = np.log1p(mat)
        mat = np.nan_to_num(mat, nan=0.0)

        count = 0
        for i in range(0, mat.shape[0] - window_bins + 1, stride):
            w = mat[i:i + window_bins, i:i + window_bins]

            if np.count_nonzero(w) / w.size < 0.05:
                continue

            w = (w - np.mean(w)) / (np.std(w) + 1e-6)
            samples.append(w.astype(np.float32)[None, ...])

            start = i * resolution
            end = (i + window_bins) * resolution
            coords.append((chrom, start, end))

            count += 1

        print(f"    Kept {count} windows from {chrom}")
    return samples, coords


# ------------------------------- filesystem helpers -------------------------------------------

def discover_coolers(roots):
    """
    Recursively walk one or more root folders and yield Cooler-openable paths.
    Supports both `.mcool` (multi-res) and `.cool` (single-res).
    Returns a list of strings ready for cooler.Cooler().
    """
    found = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"[warn] not a directory: {root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".mcool"):
                    full = os.path.join(dirpath, fn)
                    found.append((full, "mcool"))
                elif fn.endswith(".cool"):
                    full = os.path.join(dirpath, fn)
                    found.append((full, "cool"))
    return found


# ------------------------------- main ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Create Hi-C tiles dataset from folders of .mcool/.cool")
    ap.add_argument("--roots", nargs="+", default=["Data/"],
                    help="One or more folders to scan recursively for .mcool/.cool")
    ap.add_argument("--resolution", type=int, default=50000,
                    help="Resolution to open for .mcool files (ignored for .cool)")
    ap.add_argument("--window-bins", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--exclude", nargs="*", default=["chrY"],
                    help="Chromosomes to skip (by name)")
    ap.add_argument("--out", default="hic_dataset_50kb.npy",
                    help="Output .npy for tiles")
    ap.add_argument("--coords-out", default="hic_window_coords.npy",
                    help="Output .npy for (chrom,start,end) coords")
    args = ap.parse_args()

    # Find inputs
    entries = discover_coolers(args.roots)
    if not entries:
        print("[error] No .mcool/.cool files found under:", ", ".join(args.roots))
        return

    print("Found inputs:")
    for p, kind in entries:
        print(f"  - {p} ({kind})")

    all_samples = []
    window_coordinates = []

    # Process each cooler
    for p, kind in entries:
        try:
            if kind == "mcool":
                clr_path = f"{p}::resolutions/{args.resolution}"
            else:  # .cool
                clr_path = p

            print(f"\nProcessing: {clr_path}")
            clr = cooler.Cooler(clr_path)

            chroms = [c for c in clr.chromnames if c not in set(args.exclude)]
            samples, coords = slice_windows_with_coords(
                clr, chroms, args.resolution, args.window_bins, args.stride
            )
            all_samples.extend(samples)
            window_coordinates.extend(coords)

        except Exception as e:
            print(f"  Failed to process {p}: {e}")

    # Save
    if all_samples:
        dataset = np.stack(all_samples)
        print("\nFinal dataset shape:", dataset.shape)
        np.save(args.out, dataset)
        np.save(args.coords_out, np.array(window_coordinates, dtype=object))
        print(f"Saved tiles to {args.out}")
        print(f"Saved coords to {args.coords_out}")
    else:
        print("\nNo valid samples were collected.")


if __name__ == "__main__":
    main()