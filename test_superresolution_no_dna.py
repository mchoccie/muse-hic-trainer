#!/usr/bin/env python3
"""
Super-Resolution Testing Script (NO DNA)
- Evaluates only the super-resolution model conditioned on REAL low-res inputs
- Also checks VAE high-res reconstruction quality
- Robust to shape issues (forces tensors to [B,1,H,W] before plotting / metrics)
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr, spearmanr

from muse_pipeline_improved import (
    TrainingConfig, MuseTrainer, HiCDataset, custom_collate, ensure_coord_tuples
)

# ------------------------------- utils ---------------------------------

def as_b1hw(t: torch.Tensor) -> torch.Tensor:
    """Force tensor to shape [B,1,H,W]. Accepts [H,W], [1,H,W], [B,H,W], [B,1,H,W]."""
    if t.ndim == 2:                       # H, W
        return t.unsqueeze(0).unsqueeze(0)
    if t.ndim == 3:
        B, A, B_or_W = t.shape
        # If looks like [B,H,W], add channel
        if B_or_W == t.shape[-1] and A != 1:
            return t.unsqueeze(1)
        # Else assume [C,H,W] and add batch
        return t.unsqueeze(0)
    if t.ndim == 4:
        # [B,C,H,W] or already [B,1,H,W]
        if t.shape[1] == 1:
            return t
        # If accidentally [B,H,W,C], try to fix
        if t.shape[-1] == 1:
            return t.permute(0, 3, 1, 2).contiguous()
        # Otherwise squeeze to single channel by taking first chan
        return t[:, :1]
    raise ValueError(f"Unexpected tensor rank {t.ndim} for as_b1hw")

def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    """x is [H,W] or [1,H,W]; return [H,W] numpy."""
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    return x.numpy()

def safe_metrics(real_hw: np.ndarray, pred_hw: np.ndarray):
    """Compute PSNR, SSIM, Pearson, Spearman safely."""
    r = real_hw
    p = pred_hw
    rng = float(r.max() - r.min()) if float(r.max() - r.min()) > 0 else 1.0

    psnr_v = psnr(r, p, data_range=rng)
    ssim_v = ssim(r, p, data_range=rng)

    r_f = r.reshape(-1)
    p_f = p.reshape(-1)

    # handle constant arrays for correlations
    if np.std(r_f) == 0 or np.std(p_f) == 0:
        pear = 0.0
        spear = 0.0
    else:
        pear = pearsonr(r_f, p_f)[0]
        spear = spearmanr(r_f, p_f)[0]

    return psnr_v, ssim_v, pear, spear

def save_sr_panel(lr, hr, hr_rec, sr, coords, batch_idx, outdir="sr_eval_2"):
    """
    All tensors are [B,1,H,W] (enforced before calling).
    Saves 2 figures:
      - Column comparison (LR real, HR real, HR VAE recon, SR result)
      - SR vs HR only
    """
    os.makedirs(outdir, exist_ok=True)
    B = min(4, lr.size(0))

    fig, ax = plt.subplots(4, B, figsize=(5*B, 16))
    for i in range(B):
        lr_img = to_numpy_img(lr[i, 0])
        hr_img = to_numpy_img(hr[i, 0])
        rc_img = to_numpy_img(hr_rec[i, 0])
        sr_img = to_numpy_img(sr[i, 0])

        vmin = float(hr_img.min())
        vmax = float(hr_img.max())

        ax[0, i].imshow(lr_img, cmap="Reds")
        ax[0, i].set_title(f"Real Low-Res\n{coords[i]}")
        ax[0, i].axis("off")

        # upsample LR for visual context (nearest)
        lr_up = torch.tensor(lr_img)[None, None].float()
        lr_up = torch.nn.functional.interpolate(lr_up, size=hr_img.shape, mode="nearest")[0,0].numpy()
        ax[1, i].imshow(lr_up, cmap="Reds")
        ax[1, i].set_title("Low-Res (↑ to HR size)")
        ax[1, i].axis("off")

        ax[2, i].imshow(rc_img, cmap="Reds", vmin=vmin, vmax=vmax)
        ax[2, i].set_title("VAE Recon (HR)")
        ax[2, i].axis("off")

        ax[3, i].imshow(sr_img, cmap="Reds", vmin=vmin, vmax=vmax)
        ax[3, i].set_title("Super-Resolved (MaskGIT)")
        ax[3, i].axis("off")

    plt.tight_layout()
    figpath = os.path.join(outdir, f"sr_panel_batch_{batch_idx}.png")
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close()

    # SR vs HR only
    fig, ax = plt.subplots(2, B, figsize=(5*B, 8))
    for i in range(B):
        hr_img = to_numpy_img(hr[i, 0])
        sr_img = to_numpy_img(sr[i, 0])
        vmin = float(hr_img.min())
        vmax = float(hr_img.max())

        ax[0, i].imshow(hr_img, cmap="Reds", vmin=vmin, vmax=vmax)
        ax[0, i].set_title("Real High-Res")
        ax[0, i].axis("off")

        ax[1, i].imshow(sr_img, cmap="Reds", vmin=vmin, vmax=vmax)
        ax[1, i].set_title("Super-Resolved")
        ax[1, i].axis("off")

    plt.tight_layout()
    figpath2 = os.path.join(outdir, f"sr_vs_hr_batch_{batch_idx}.png")
    plt.savefig(figpath2, dpi=150, bbox_inches="tight")
    plt.close()

def eval_superres(trainer, num_batches=4, timesteps=18):
    """
    Evaluate super-resolution:
      - SR model with REAL LR conditioning (no DNA)
      - HR VAE reconstruction for reference
    """
    trainer.maskgit_high.eval()

    # choose split to evaluate on
    ds = trainer.val_ds if hasattr(trainer, "val_ds") else trainer.train_ds

    dl = DataLoader(
        ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=custom_collate
    )

    all_psnr, all_ssim, all_pear, all_spear = [], [], [], []

    device = trainer.device

    with torch.no_grad():
        for b, (lr, hr, coords) in enumerate(dl):
            if b >= num_batches:
                break

            coords = ensure_coord_tuples(coords)

            lr = lr.to(device).float()      # expect [B,1,256,256]
            hr = hr.to(device).float()      # expect [B,1,512,512]

            # 1) VAE HR reconstruction
            fmap, _, _ = trainer.vae_high.encode(hr)    # fmap: latent fmap
            hr_rec = trainer.vae_high.decode(fmap)      # -> [B,1,512,512]

            # 2) SR with REAL LR conditioning (no DNA)
            sr = trainer.maskgit_high.generate(
                cond_images=lr,
                timesteps=timesteps
            )

            # Coerce shapes
            lr     = as_b1hw(lr)
            hr     = as_b1hw(hr)
            hr_rec = as_b1hw(hr_rec)
            sr     = as_b1hw(sr)

            # metrics
            for i in range(min(lr.size(0), hr.size(0), sr.size(0))):
                hr_real = to_numpy_img(hr[i, 0])
                sr_pred = to_numpy_img(sr[i, 0])

                psnr_v, ssim_v, pear_v, spear_v = safe_metrics(hr_real, sr_pred)
                all_psnr.append(psnr_v)
                all_ssim.append(ssim_v)
                all_pear.append(pear_v)
                all_spear.append(spear_v)

            save_sr_panel(lr, hr, hr_rec, sr, coords, b)

    print("\n=== Super-Resolution Summary (No DNA) ===")
    if all_psnr:
        print(f"PSNR   : {np.mean(all_psnr):.3f} ± {np.std(all_psnr):.3f}")
        print(f"SSIM   : {np.mean(all_ssim):.3f} ± {np.std(all_ssim):.3f}")
        print(f"Pearson: {np.mean(all_pear):.3f} ± {np.std(all_pear):.3f}")
        print(f"Spearman: {np.mean(all_spear):.3f} ± {np.std(all_spear):.3f}")
    else:
        print("No metrics collected (empty loader or errors).")

def load_for_test(checkpoint_path: str, cfg: TrainingConfig) -> MuseTrainer:
    """
    Construct trainer and load checkpoint. Accepts both old and new key names.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    trainer = MuseTrainer(cfg)  # builds models + loads datasets
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # flexible key handling
    def _load(module, keys):
        for k in keys:
            if k in ckpt:
                module.load_state_dict(ckpt[k])
                return True
        return False

    ok1 = _load(trainer.maskgit_low,  ["maskgit_low",  "maskgit_lowres_state_dict",  "maskgit_low_state_dict"])
    ok2 = _load(trainer.maskgit_high, ["maskgit_high", "maskgit_highres_state_dict", "maskgit_high_state_dict"])

    if not (ok1 and ok2):
        print("[Warn] Could not find expected 'maskgit_*' keys; proceeding with randomly init'd modules.")

    # (optionally) load optimizers if present
    if "opt_low" in ckpt:
        trainer.opt_low.load_state_dict(ckpt["opt_low"])
    if "opt_high" in ckpt:
        trainer.opt_high.load_state_dict(ckpt["opt_high"])

    trainer.current_epoch = ckpt.get("epoch", 0)
    trainer.global_step  = ckpt.get("global_step", 0)
    trainer.best_val_loss = ckpt.get("best_val_loss", float("inf"))

    print(f"Loaded at epoch={trainer.current_epoch}, step={trainer.global_step}, best_val={trainer.best_val_loss:.4f}")
    return trainer

# ------------------------------- main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpointbest.pt")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--num-batches", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=18)
    args = ap.parse_args()

    ckpt = args.checkpoint if args.epoch is None else f"checkpoint_epoch_{args.epoch}.pt"
    if not os.path.exists(ckpt):
        print(f"[Error] Checkpoint not found: {ckpt}")
        return

    cfg = TrainingConfig()
    trainer = load_for_test(ckpt, cfg)

    # important: ensure eval mode + no grads
    trainer.maskgit_low.eval()
    trainer.maskgit_high.eval()
    torch.set_grad_enabled(False)

    eval_superres(trainer, num_batches=args.num_batches, timesteps=args.timesteps)
    print("\n=== Super-Resolution Testing Complete (No DNA) ===")

if __name__ == "__main__":
    main()
