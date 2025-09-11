#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

def _squeeze_to_2d(x: np.ndarray) -> np.ndarray:
    """Accept (H,W), (1,H,W), (H,W,1), (C,H,W) and return (H,W)."""
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        if x.shape[0] == 1:      # (1,H,W)
            return x[0]
        if x.shape[-1] == 1:     # (H,W,1)
            return x[..., 0]
        return x[0]              # take first channel
    raise ValueError(f"Unsupported sample shape: {x.shape}")

def _load_sample(path: str, idx: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        return arr
    return _squeeze_to_2d(arr[idx])

def _load_mask(path: str, idx: int, target_shape) -> np.ndarray | None:
    if not path:
        return None
    m = np.load(path, mmap_mode="r")
    m = _squeeze_to_2d(m[idx])
    if m.shape != target_shape:
        raise ValueError(f"Mask shape {m.shape} != data shape {target_shape}")
    return m

def _coords_label(coords_obj, idx, bins):
    try:
        c = np.load(coords_obj, allow_pickle=True)
        chrom, start, end = c[idx]
        mb = lambda b: f"{b/1e6:.1f}"
        res_bp = (end - start) / bins if bins > 0 else None
        res_str = f" (≈{res_bp/1000:.0f} kb/bin)" if res_bp else ""
        return f"{chrom}:{int(start):,}-{int(end):,}{res_str}", (chrom, int(start), int(end))
    except Exception:
        return None, None

def _mb_formatter(start_bp, end_bp, bins):
    def _fmt(x, pos):
        # x is in bin units
        bp = start_bp + (x / max(bins-1, 1)) * (end_bp - start_bp)
        return f"{bp/1e6:.1f}"
    return FuncFormatter(_fmt)

def main():
    ap = argparse.ArgumentParser(description="Simple Hi-C style viewer for .npy tiles")
    ap.add_argument("--file", required=True, help=".npy file (H,W) or (N,1,H,W)/(N,H,W)")
    ap.add_argument("--idx", type=int, default=0, help="sample index (if file is a stack)")
    ap.add_argument("--coords", help="coords .npy (object tuples (chrom,start,end)) for axis labels")
    ap.add_argument("--mask", help="optional mask .npy to overlay (same indexing)")
    ap.add_argument("--triangular", action="store_true", help="show only upper triangle (transparent lower)")
    ap.add_argument("--log1p", action="store_true", help="apply log1p before display (recommended for counts)")
    ap.add_argument("--pclip", type=float, default=99.5, help="percentile clip for vmax (default 99.5)")
    ap.add_argument("--vmin", type=float, default=None, help="fixed vmin (overrides pclip lower)")
    ap.add_argument("--vmax", type=float, default=None, help="fixed vmax (overrides pclip upper)")
    ap.add_argument("--cmap", default="Reds", help="matplotlib colormap")
    ap.add_argument("--out", default=None, help="save PNG here (omit to show)")
    args = ap.parse_args()

    M = _load_sample(args.file, args.idx).astype(np.float32)
    H, W = M.shape
    assert H == W, "expects square tile"

    # transform for visualization
    if args.log1p:
        M = np.log1p(M)

    # percentile-based clipping for display
    if args.vmin is None or args.vmax is None:
        lo = np.percentile(M, 100 - args.pclip) if args.vmin is None else args.vmin
        hi = np.percentile(M, args.pclip)       if args.vmax is None else args.vmax
    else:
        lo, hi = args.vmin, args.vmax
    norm = Normalize(vmin=float(lo), vmax=float(hi), clip=True)

    # optional triangular: hide lower triangle with transparent NaNs
    if args.triangular:
        M_plot = M.copy()
        tril = np.tril_indices(H, k=-1)
        M_plot[tril] = np.nan
    else:
        M_plot = M

    # optional mask overlay
    mask = _load_mask(args.mask, args.idx, (H, W)) if args.mask else None

    # title + genomic labels
    coords_title, coords_tuple = (None, None)
    if args.coords:
        coords_title, coords_tuple = _coords_label(args.coords, args.idx, H)

    fig = plt.figure(figsize=(9, 8))
    ax = plt.gca()

    # make NaNs transparent
    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(alpha=0.0)

    im = ax.imshow(M_plot, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("value (log1p)" if args.log1p else "value", rotation=90)

    if mask is not None:
        invalid = (mask <= 0).astype(np.float32)
        ax.imshow(invalid, cmap="gray", alpha=invalid * 0.35, origin="upper", interpolation="nearest")

    # axes cosmetics
    ax.set_aspect("equal")
    if coords_tuple:
        chrom, start, end = coords_tuple
        ax.xaxis.set_major_formatter(_mb_formatter(start, end, H))
        ax.yaxis.set_major_formatter(_mb_formatter(start, end, H))
        ax.set_xlabel("Genomic position (Mb)")
        ax.set_ylabel("Genomic position (Mb)")
        title = f"{os.path.basename(args.file)}[{args.idx}]  —  {chrom}:{start:,}-{end:,}"
    else:
        ax.set_xlabel("bin")
        ax.set_ylabel("bin")
        title = f"{os.path.basename(args.file)}[{args.idx}]"

    if args.triangular:
        title += "  (upper triangle)"

    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=250, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()