#!/usr/bin/env python3
"""
Super-Resolution Testing Script (Single VAE, No DNA)
- Matches the dataset-builder normalization:
    Balanced (if your dataloader gives balanced counts)
    -> O/E by distance stratum (within-window)
    -> asinh
    -> per-stratum z-score (within-window)
- You can either:
    (A) Assume tiles are already preprocessed (no-op), or
    (B) Apply the above pipeline at eval time for LR & HR
- Also supports mapping to model-expected ranges: same / 0..1 / -1..1
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr, spearmanr

from muse_pipeline_downsampled import (
    TrainingConfig, MuseTrainer, custom_collate, ensure_coord_tuples
)

# ------------------------- exact normalization used to build data -------------------------

def _stratum_indices(n: int, k: int):
    r = np.arange(n - k)
    c = r + k
    upper = (r, c)
    if k == 0:
        return [upper]
    lower = (c, r)
    return [upper, lower]

def oe_asinh_stratum_zscore_tile(H: np.ndarray, Hm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    H = H.astype(np.float32, copy=False)
    Hm = (Hm.astype(np.float32, copy=False) > 0).astype(np.float32)
    n = H.shape[0]

    # 1) expected per stratum within this window
    expected = np.zeros(n, dtype=np.float32)
    for k in range(n):
        vals = []
        for (rr, cc) in _stratum_indices(n, k):
            mvalid = Hm[rr, cc] > 0
            if mvalid.any():
                vals.append(H[rr, cc][mvalid])
        expected[k] = float(np.mean(np.concatenate(vals))) if vals else 0.0

    # 2) O/E
    OE = np.zeros_like(H, dtype=np.float32)
    for k in range(n):
        denom = expected[k] if expected[k] > eps else 1.0
        for (rr, cc) in _stratum_indices(n, k):
            OE[rr, cc] = H[rr, cc] / denom

    # 3) asinh
    T = np.arcsinh(OE).astype(np.float32)

    # 4) per-stratum z within window
    Z = np.zeros_like(T, dtype=np.float32)
    for k in range(n):
        vals = []
        idxs = []
        for (rr, cc) in _stratum_indices(n, k):
            mvalid = (Hm[rr, cc] > 0)
            if mvalid.any():
                vals.append(T[rr, cc][mvalid])
                idxs.append((rr[mvalid], cc[mvalid]))
        if not vals:
            continue
        cat = np.concatenate(vals)
        mu = float(cat.mean())
        sd = float(cat.std(ddof=0))
        if sd < eps:
            for (rrv, ccv) in idxs:
                Z[rrv, ccv] = T[rrv, ccv] - mu
        else:
            for (rrv, ccv) in idxs:
                Z[rrv, ccv] = (T[rrv, ccv] - mu) / sd

    Z[Hm == 0] = 0.0
    return Z

def preprocess_counts_batch_oe_asinh_stratz(X: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    X, masks: [B,1,H,W] or [B,H,W] tensors (values on CPU/GPU ok).
    Returns same shape tensor (float32) in the normalized space.
    """
    x = X.detach().cpu().float()
    m = masks.detach().cpu().float()
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
        m = m[:, 0]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {X.shape}")

    out = []
    for i in range(x.shape[0]):
        z = oe_asinh_stratum_zscore_tile(x[i].numpy(), (m[i] > 0).numpy())
        out.append(torch.from_numpy(z))
    out = torch.stack(out, dim=0)  # [B,H,W]
    return out[:, None].to(X.device)  # back to [B,1,H,W]

# ------------------------------- helpers ---------------------------------

def as_b1hw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 3:
        if t.shape[0] != 1:
            return t.unsqueeze(1)
        return t.unsqueeze(0)
    if t.ndim == 4:
        if t.shape[1] == 1:
            return t
        if t.shape[-1] == 1:
            return t.permute(0, 3, 1, 2).contiguous()
        return t[:, :1]
    raise ValueError(f"Unexpected tensor rank {t.ndim}")


def _infer_fmap_size_from_maskgit(maskgit, x_like):
    """
    Return integer latent fmap size f (so seq_len = f*f).
    Uses cached size if available; otherwise infers via a 1-sample encode.
    """
    f = None
    if hasattr(maskgit.vae, "get_encoded_fmap_size"):
        f = maskgit.vae.get_encoded_fmap_size(maskgit.image_size)
    if f is None:
        with torch.no_grad():
            fmap, _, _ = maskgit.vae.encode(x_like[:1])
        f = int(fmap.shape[-1])
    return f

def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return x.numpy()

def safe_metrics(real_hw: np.ndarray, pred_hw: np.ndarray):
    r, p = real_hw, pred_hw
    rng = float(r.max() - r.min()) if float(r.max() - r.min()) > 0 else 1.0
    psnr_v = psnr(r, p, data_range=rng)
    ssim_v = ssim(r, p, data_range=rng)
    r_f, p_f = r.reshape(-1), p.reshape(-1)
    if np.std(r_f) == 0 or np.std(p_f) == 0:
        return psnr_v, ssim_v, 0.0, 0.0
    return psnr_v, ssim_v, pearsonr(r_f, p_f)[0], spearmanr(r_f, p_f)[0]

# --- replace your save_sr_panel(...) with this ---
def save_three_row_panel(lr, hr, sr, coords, batch_idx, outdir="test_normal_generate_after_changes"):
    """
    Rows:
      0: Predicted (MaskGIT), title includes Pearson r vs HR
      1: Low-Depth (upsampled for display)
      2: High-Depth (ground truth)
    """
    os.makedirs(outdir, exist_ok=True)
    B = min(4, lr.size(0))

    fig, ax = plt.subplots(3, B, figsize=(5*B, 12))
    for i in range(B):
        lr_img = to_numpy_img(lr[i, 0])
        hr_img = to_numpy_img(hr[i, 0])
        sr_img = to_numpy_img(sr[i, 0])

        # lock color scale to HR
        vmin, vmax = float(hr_img.min()), float(hr_img.max())

        # Pearson r between SR and HR (flattened; guard zero-variance)
        r_val = np.nan
        r_flat, s_flat = hr_img.reshape(-1), sr_img.reshape(-1)
        if (np.std(r_flat) > 0) and (np.std(s_flat) > 0):
            r_val = pearsonr(r_flat, s_flat)[0]

        # row 0: predicted
        ax[0, i].imshow(sr_img, cmap="Reds", vmin=vmin, vmax=vmax)
        title = f"Predicted (r={r_val:.3f})" if r_val == r_val else "Predicted (r=nan)"
        ax[0, i].set_title(title)
        ax[0, i].axis("off")

        # row 1: low-depth (upsampled for side-by-side)
        lr_up = torch.tensor(lr_img)[None, None].float()
        lr_up = torch.nn.functional.interpolate(lr_up, size=hr_img.shape, mode="nearest")[0, 0].numpy()
        ax[1, i].imshow(lr_up, cmap="Reds", vmin=vmin, vmax=vmax)
        ax[1, i].set_title(f"Low-Depth (↑)\n{coords[i]}")
        ax[1, i].axis("off")

        # row 2: high-depth (ground truth)
        ax[2, i].imshow(hr_img, cmap="Reds", vmin=vmin, vmax=vmax)
        ax[2, i].set_title("High-Depth (GT)")
        ax[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"panel_batch_{batch_idx}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

def print_stats(tag, x: torch.Tensor):
    x = as_b1hw(x)
    arr = x[:, 0].detach().cpu().numpy()
    print(f"{tag:>10s}  shape={list(x.shape)}  min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}  std={arr.std():.3f}")

# ------------------------------- evaluation ---------------------------------

def maybe_map_to_expected_range(x: torch.Tensor, expects: str) -> torch.Tensor:
    """
    Map normalized tiles to what the VAE/MaskGIT expect:
      - "same": do nothing (assumes training used the same normalized space)
      - "01"  : map from current space (assumed z-scored or similar) to [0,1] via sigmoid-ish clamp
      - "m1p1": map to [-1,1]
    NOTE: If you actually trained on the oe/asinh/stratz space, use --expects same.
    These mappings are conservative; prefer retraining or using the exact training scaler when possible.
    """
    if expects == "same":
        return x
    if expects == "01":
        # squash to 0..1 without blowing up outliers
        return torch.clamp((x - x.mean(dim=[2,3], keepdim=True)) / (x.std(dim=[2,3], keepdim=True) + 1e-6) * 0.15 + 0.5, 0.0, 1.0)
    if expects == "m1p1":
        return torch.clamp((x - x.mean(dim=[2,3], keepdim=True)) / (x.std(dim=[2,3], keepdim=True) + 1e-6) * 0.3, -1.0, 1.0)
    raise ValueError(f"expects must be one of ['same','01','m1p1'], got {expects}")

@torch.no_grad()
def seeded_edit_generate(
    maskgit,
    seed_images,           # [B,1,512,512]
    cond_images,           # [B,1,512,512]
    timesteps=28,
    # allow both names; if edit_frac is passed, it’s treated as start
    edit_frac_start=0.3,
    edit_frac_end=0.7,
    edit_frac=None,        # alias for edit_frac_start
    focus="band",
    band_frac=0.10,
    # allow both names; if topp is given we map to topk
    topk=0.95,
    topp=None,
    temperature=0.9,
    use_entropy_gate=True,
    entropy_keep_frac=0.5,
    fmap_size=None         # allow callers to pass it; otherwise infer
):
    # ---- arg aliasing ----
    if (edit_frac is not None) and (edit_frac_start == 0.3):
        edit_frac_start = float(edit_frac)
    if (topp is not None) and (topk == 0.95):
        topk = float(topp)

    # 1) seed ids
    _, ids_seed, _ = maskgit.vae.encode(seed_images)  # [B,f,f]
    B, f, _ = ids_seed.shape
    S = f * f
    device = ids_seed.device

    # ensure fmap_size for generate()
    if fmap_size is None:
        fmap_size = _infer_fmap_size_from_maskgit(maskgit, seed_images)

    # 2) editable mask
    editable = torch.ones(B, S, dtype=torch.bool, device=device)

    # diagonal band protection (optional)
    if focus == "band":
        g = torch.arange(f, device=device)
        rr = g[:, None].expand(f, f)
        cc = g[None, :].expand(f, f)
        band = (rr - cc).abs() <= max(1, int(round(band_frac * f)))
        diag_mask = band.view(1, -1)
    else:
        diag_mask = torch.zeros(1, S, dtype=torch.bool, device=device)

    # 3) entropy gate (optional)
    if use_entropy_gate:
        ids_flat = ids_seed.view(B, S)
        cond_ids = None
        if maskgit.resize_image_for_cond_image:
            _, cond_ids, _ = maskgit.cond_vae.encode(cond_images)
            cond_ids = cond_ids.view(B, -1)
        logits = maskgit.transformer.forward_with_cond_scale(
            ids_flat,
            text_embeds=None,
            conditioning_token_ids=cond_ids,
            self_cond_embed=None,
            cond_scale=1.0,
            return_embed=False
        )
        probs = logits.softmax(-1)
        entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum(-1)  # [B,S]
        k = max(1, int(round(entropy.shape[1] * entropy_keep_frac)))
        thresh = torch.kthvalue(entropy, entropy.shape[1] - k, dim=1).values[:, None]
        entropy_gate = entropy >= thresh
        editable &= entropy_gate

    # 4) build immutable+edit masks
    immutable_mask = diag_mask.expand(B, S).clone()
    edit_mask = editable.clone()

    # thin edit mask to ~edit_frac_start density
    desired_frac = float(edit_frac_start)
    num_desired = (edit_mask.sum(dim=1).float() * desired_frac).long().clamp(min=1)
    perm = torch.rand(B, S, device=device).argsort(dim=-1)
    rank = torch.zeros_like(perm)
    rank.scatter_(1, perm, torch.arange(S, device=device).expand(B, S))
    edit_mask = (rank < num_desired[:, None]) & edit_mask

    # 5) single-call generate with masks
    out = maskgit.generate(
        cond_images=cond_images,
        timesteps=timesteps,
        temperature=temperature,
        topk_filter_thres=topk,
        start_ids=ids_seed,                         # [B,f,f]
        immutable_mask=immutable_mask,              # [B,S] bool
        edit_mask=edit_mask,                        # [B,S] bool
        fmap_size=fmap_size,
        cond_scale=1.0
    )
    return out

@torch.no_grad()
def eval_superres(
    trainer,
    num_batches=4,
    timesteps=18,
    apply_oe=False,
    assume_preprocessed=False,
    expects="same",
    # --- seeded-edit params ---
    seeded=False,
    edit_frac=0.6,          # <— keep this name
    edit_frac_start=None,   # new parameter
    edit_frac_end=None,     # new parameter
    focus="all",
    band_frac=0.10,
    temperature=0.9,
    topp=0.95,
    topk=None,              # new parameter
    cond_scale=None
):
    # Handle parameter aliasing
    if edit_frac_start is not None:
        edit_frac = edit_frac_start
    if topk is not None:
        topp = topk
    
    trainer.maskgit.eval()
    ds = trainer.val_ds if hasattr(trainer, "val_ds") else trainer.train_ds
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate)

    all_psnr, all_ssim, all_pear, all_spear = [], [], [], []
    device = trainer.device

    with torch.no_grad():
        for b, (lr, hr, coords) in enumerate(dl):
            if b >= num_batches:
                break
            coords = ensure_coord_tuples(coords)
            lr, hr = lr.to(device).float(), hr.to(device).float()

            lr = as_b1hw(lr)
            hr = as_b1hw(hr)

            if apply_oe and not assume_preprocessed:
                lr_mask = torch.isfinite(lr) & (lr == lr)
                hr_mask = torch.isfinite(hr) & (hr == hr)
                lr = preprocess_counts_batch_oe_asinh_stratz(lr, lr_mask)
                hr = preprocess_counts_batch_oe_asinh_stratz(hr, hr_mask)

            lr_in = maybe_map_to_expected_range(lr, expects)
            hr_in = maybe_map_to_expected_range(hr, expects)

            if b == 0:
                print_stats("LR(in)", lr_in)
                print_stats("HR(in)", hr_in)

            # --- Generate SR ---
            f = _infer_fmap_size_from_maskgit(trainer.maskgit, lr_in)
            if seeded:
                sr = seeded_edit_generate(
                    trainer.maskgit,
                    seed_images=lr_in,
                    cond_images=lr_in,
                    timesteps=timesteps,
                    edit_frac=edit_frac,     # alias → start
                    edit_frac_start=edit_frac_start,
                    edit_frac_end=edit_frac_end,
                    focus=focus,
                    band_frac=band_frac,
                    temperature=temperature,
                    topp=topp,               # will map to topk inside
                    fmap_size=f
                )
            else:
                sr = trainer.maskgit.generate(
                    cond_images=lr_in,
                    timesteps=timesteps,
                    temperature=temperature,
                    topk_filter_thres=topp,  # your MaskGit uses 'topk_filter_thres'
                    fmap_size=f
                )

            # Compare in the same normalized domain
            comp_hr = hr_in
            comp_sr = sr

            if b == 0:
                print_stats("SR(out)", comp_sr)

            # metrics + per-tile Pearson prints
            lr_b, hr_b, sr_b = map(as_b1hw, (lr_in, comp_hr, comp_sr))
            for i in range(min(lr_b.size(0), hr_b.size(0), sr_b.size(0))):
                lr_real = to_numpy_img(lr_b[i, 0])
                hr_real = to_numpy_img(hr_b[i, 0])
                sr_pred = to_numpy_img(sr_b[i, 0])

                psnr_v, ssim_v, pear_sr, spear_sr = safe_metrics(hr_real, sr_pred)
                all_psnr.append(psnr_v); all_ssim.append(ssim_v)
                all_pear.append(pear_sr); all_spear.append(spear_sr)

                _, _, pear_lr, spear_lr = safe_metrics(hr_real, lr_real)
                print(f"[batch {b} sample {i}] r(SR,HR)={pear_sr:.4f} | r(LR,HR)={pear_lr:.4f}")

            # three-row panel (Pred, LR↑, HR)
            save_three_row_panel(lr_b, comp_hr, comp_sr, coords, b)

    print("\n=== Super-Resolution Summary ===")
    if all_psnr:
        print(f"PSNR    : {np.mean(all_psnr):.3f} ± {np.std(all_psnr):.3f}")
        print(f"SSIM    : {np.mean(all_ssim):.3f} ± {np.std(all_ssim):.3f}")
        print(f"Pearson : {np.mean(all_pear):.3f} ± {np.std(all_pear):.3f}")
        print(f"Spearman: {np.mean(all_spear):.3f} ± {np.std(all_spear):.3f}")
    else:
        print("No metrics collected.")

def load_for_test(checkpoint_path: str, cfg: TrainingConfig) -> MuseTrainer:
    print(f"Loading checkpoint: {checkpoint_path}")
    trainer = MuseTrainer(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "maskgit" in ckpt:
        trainer.maskgit.load_state_dict(ckpt["maskgit"])
    else:
        print("[Warn] No 'maskgit' weights found in checkpoint.")

    if "opt" in ckpt:
        trainer.opt.load_state_dict(ckpt["opt"])

    trainer.current_epoch = ckpt.get("epoch", 0)
    trainer.global_step   = ckpt.get("step", 0)
    trainer.best_val_loss = ckpt.get("best_val", float("inf"))

    print(f"Loaded at epoch={trainer.current_epoch}, step={trainer.global_step}, best_val={trainer.best_val_loss:.4f}")
    return trainer

# ------------------------------- main ---------------------------------

def main():
    ap = argparse.ArgumentParser()

    # checkpoint / run control
    ap.add_argument("--checkpoint", type=str, default="checkpointbest.pt")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--num-batches", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=18)

    # seeded-editing options (start from LR tokens and only edit a subset)
    ap.add_argument("--seeded", action="store_true",
                    help="Use seeded editing (start from LR tokens) instead of classic sampling.")
    ap.add_argument("--edit-frac", type=float, default=0.6,
                    help="Initial fraction of tokens to edit (0..1) when --seeded is set.")
    ap.add_argument("--focus", type=str, default="all", choices=["all", "band"],
                    help="Where to allow edits when --seeded: 'all' tokens or off-diagonal 'band'.")
    ap.add_argument("--band-frac", type=float, default=0.10,
                    help="Diagonal half-width (fraction of latent size) to protect when focus='band'.")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="Sampling temperature for generation.")
    ap.add_argument("--topp", type=float, default=0.95,
                    help="Top-p (nucleus) sampling for generation.")
    ap.add_argument("--cond-scale", type=float, default=None,
                    help="Classifier-free guidance scale if your MaskGIT fork supports it (e.g., 2.0–4.0).")
    # NEW preferred knobs
    ap.add_argument("--edit-frac-start", type=float, default=0.30,
                    help="Starting fraction of tokens to edit (0..1) for seeded editing.")
    ap.add_argument("--edit-frac-end", type=float, default=0.70,
                    help="Ending fraction of tokens to edit (0..1) for seeded editing.")
    ap.add_argument("--topk", type=float, default=0.95,
                    help="Top-k filter threshold for sampling (0..1).")

    # normalization options
    ap.add_argument("--apply-oe-stratz", action="store_true",
                    help="Apply window-local O/E→asinh→per-stratum z-score to LR & HR before eval.")
    ap.add_argument("--assume-preprocessed", action="store_true",
                    help="Skip normalization (use if your dataset already outputs preprocessed tiles).")
    ap.add_argument("--expects", type=str, default="same", choices=["same", "01", "m1p1"],
                    help="Numeric range the VAE/MaskGIT expect. Use 'same' if trained on oe/asinh/stratz.")

    args = ap.parse_args()
    args.topk = args.topk if args.topk is not None else (args.topp if args.topp is not None else 0.95)
    args.edit_frac_start = args.edit_frac_start if args.edit_frac_start is not None else (
        args.edit_frac if args.edit_frac is not None else 0.30
    )
    args.edit_frac_end = args.edit_frac_end if args.edit_frac_end is not None else args.edit_frac_start

    ckpt = args.checkpoint if args.epoch is None else f"checkpoint_epoch_{args.epoch}.pt"
    if not os.path.exists(ckpt):
        print(f"[Error] Checkpoint not found: {ckpt}")
        return

    cfg = TrainingConfig()
    trainer = load_for_test(ckpt, cfg)
    trainer.maskgit.eval()
    torch.set_grad_enabled(False)

    # make sure your eval_superres signature accepts these kwargs
    eval_superres(
        trainer,
        num_batches=args.num_batches,
        timesteps=args.timesteps,
        apply_oe=args.apply_oe_stratz,
        assume_preprocessed=args.assume_preprocessed,
        expects=args.expects,
        seeded=args.seeded,
        edit_frac_start=args.edit_frac_start,
        edit_frac_end=args.edit_frac_end,
        focus=args.focus,
        band_frac=args.band_frac,
        temperature=args.temperature,
        topk=args.topk,
        cond_scale=args.cond_scale
    )

    print("\n=== Super-Resolution Testing Complete ===")

if __name__ == "__main__":
    main()
