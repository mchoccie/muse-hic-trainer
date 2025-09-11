#!/usr/bin/env python3
import argparse, numpy as np, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low",  required=True, help="lowdepth_dataset.npy (N,1,H,W)")
    ap.add_argument("--high", required=True, help="highdepth_dataset.npy (N,1,H,W)")
    ap.add_argument("--out",  default="lr_hr.png", help="output image path")
    ap.add_argument("--idx",  type=int, default=0, help="tile index to plot")
    args = ap.parse_args()

    lo = np.load(args.low,  mmap_mode="r")
    hi = np.load(args.high, mmap_mode="r")

    i = max(0, min(int(args.idx), lo.shape[0]-1))
    L = lo[i, 0]
    H = hi[i, 0]

    # lock color scale to HR so LR isn't auto-rescaled
    vmin, vmax = float(H.min()), float(H.max())

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(L, cmap="Reds", vmin=vmin, vmax=vmax); ax[0].axis("off"); ax[0].set_title("Low-Depth")
    ax[1].imshow(H, cmap="Reds", vmin=vmin, vmax=vmax); ax[1].axis("off"); ax[1].set_title("High-Depth")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()