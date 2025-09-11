# muse_pipeline_improved.py  — single-VAE, same-size conditioning (512x512)

import os, math, time, warnings, sys
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# --- local / repo imports ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'muse-maskgit-pytorch'))
from muse_maskgit_pytorch import MaskGit, MaskGitTransformer
from vqgan_vae_hic_weighted import VQGanVAE
from muse_maskgit_pytorch.improved_dna_encoders import create_dna_encoder


# =============================== utils =======================================

def exists(x): return x is not None

def psnr(x, y, max_val=None):
    # x,y: (B,1,H,W). For normalized z-scores, pick a stable max_val just for tracking.
    mse = F.mse_loss(x, y)
    if max_val is None:
        max_val = 6.0
    return 10 * math.log10((max_val ** 2) / (mse.item() + 1e-12))

@torch.no_grad()
def codebook_perplexity(vq: VQGanVAE, batch):
    # batch: (B,1,H,W)
    _, ids, _ = vq.encode(batch)     # (B, f, f)
    K = vq.codebook_size
    hist = torch.bincount(ids.view(-1), minlength=K).float()
    p = hist / hist.sum().clamp_min(1)
    perp = torch.exp(-(p * (p.clamp_min(1e-12)).log()).sum())
    used = (hist > 0).float().sum().item()
    return perp.item(), int(used)

def warmup_cosine(step, warmup, total_steps, min_lr=1e-6):
    if step < warmup:
        return max(min_lr, (step + 1) / max(1, warmup))
    progress = (step - warmup) / max(1, total_steps - warmup)
    return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

def ensure_coord_tuples(coords):
    out = []
    for c in coords:
        if isinstance(c, tuple):
            out.append(c)
        elif isinstance(c, (list, np.ndarray)):
            flat = []
            for x in c:
                if isinstance(x, np.generic):
                    flat.append(int(x))
                else:
                    flat.append(x)
            assert len(flat) == 3, f"Bad coord (len {len(flat)}): {c}"
            out.append(tuple(flat))
        else:
            raise TypeError(f"Bad coord type {type(c)}: {c}")
    return out

@torch.no_grad()
def build_mask_like_forward(maskgit, images_or_ids):
    """
    Recreate MaskGit.forward() masking:
      - tokenizes if floats
      - uses maskgit.noise_schedule(t)
      - at least 1 token masked
    Returns (ids_flat, mask_bool, f)
    """
    if images_or_ids.dtype == torch.float:
        _, ids, _ = maskgit.vae.encode(images_or_ids)  # (B, f, f)
    else:
        ids = images_or_ids

    B, f, _ = ids.shape
    ids_flat = ids.view(B, -1)
    S = ids_flat.size(1)
    device = ids_flat.device

    t = torch.rand((B,), device=device)
    mask_ratio = maskgit.noise_schedule(t)
    if not torch.is_tensor(mask_ratio):
        mask_ratio = torch.tensor(mask_ratio, device=device).expand(B)
    k = (S * mask_ratio).round().clamp(min=1).long()

    perm = torch.rand((B, S), device=device).argsort(dim=-1)
    mask_bool = perm < k[:, None]
    return ids_flat, mask_bool, f

def visualize_mask_once(maskgit, img, step, save_dir="mask_visualizations"):
    """
    img: (B,1,H,W) float
    Shows original, binary mask (white=masked), and overlay.
    """
    os.makedirs(save_dir, exist_ok=True)
    ids_flat, mask_bool, f = build_mask_like_forward(maskgit, img)

    B, S = mask_bool.shape
    H, W = img.shape[-2:]

    m = mask_bool[0].view(f, f).float()
    m_up = F.interpolate(m[None, None], size=(H, W), mode="nearest")[0, 0]

    x = img[0, 0].detach().cpu().numpy()
    mnp = m_up.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(x, cmap="Reds", vmin=x.min(), vmax=x.max()); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(mnp, cmap="gray", vmin=0, vmax=1);          ax[1].set_title("Mask (white = masked)"); ax[1].axis("off")

    overlay = x.copy()
    overlay[mnp > 0.5] = overlay[mnp > 0.5] * 1.2
    ax[2].imshow(overlay, cmap="Reds", vmin=x.min(), vmax=x.max()); ax[2].set_title("Overlay"); ax[2].axis("off")

    pct = 100.0 * mask_bool[0].float().mean().item()
    fig.suptitle(f"Step {step} — masked {mask_bool[0].sum().item()}/{S} tokens ({pct:.1f}%)")
    path = os.path.join(save_dir, f"mask_step_{step}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"✓ saved {path}")

@torch.no_grad()
def save_vae_recon_panel(x, rec, step, name, outdir="vae_recons"):
    """
    x, rec: [B,1,H,W] tensors in the SAME range fed to the VAE (e.g., [0,1] or [-1,1]).
    Saves a side-by-side panel of up to 4 examples and returns the file path.
    """
    os.makedirs(outdir, exist_ok=True)
    x_np  = x.detach().cpu().numpy()
    r_np  = rec.detach().cpu().numpy()
    B = min(4, x_np.shape[0])

    fig, ax = plt.subplots(2, B, figsize=(4*B, 6))
    for i in range(B):
        inp  = x_np[i, 0]
        out  = r_np[i, 0]
        vmin, vmax = float(inp.min()), float(inp.max())
        ax[0, i].imshow(inp, cmap="Reds", vmin=vmin, vmax=vmax); ax[0, i].set_title("Input");  ax[0, i].axis("off")
        ax[1, i].imshow(out, cmap="Reds", vmin=vmin, vmax=vmax); ax[1, i].set_title("Recon");  ax[1, i].axis("off")
    fig.suptitle(f"{name} — step {step}")
    plt.tight_layout()
    path = os.path.join(outdir, f"{name}_step_{step}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"✓ saved {path}")
    return path


# =============================== dataset ======================================

def flip_both(x):        return torch.flip(x, dims=[-2, -1])
def transpose_diag(x):   return x.transpose(-2, -1)
def roll_same(x, k):     return torch.roll(torch.roll(x, shifts=k, dims=-1), shifts=k, dims=-2)
def intensity_jitter(x, scale_range=(0.95, 1.05), bias_range=(-0.05, 0.05)):
    s = torch.empty(1, device=x.device).uniform_(*scale_range).item()
    b = torch.empty(1, device=x.device).uniform_(*bias_range).item()
    return x * s + b

class HiCDataset(Dataset):
    """
    Expects arrays:
      - lowdepth:  (N, 1, 512, 512)  normalized low-depth
      - highdepth: (N, 1, 512, 512)  normalized high-depth (target)
      - coords   : list[ (chrom, start_bp, end_bp) ]
    Augmentations are symmetry-safe for Hi-C.
    """
    def __init__(self, low_np, high_np, coords, augment=True, repeat_factor=1, max_shift_bins=2):
        self.low_np   = low_np
        self.high_np  = high_np
        self.coords   = coords
        self.augment  = augment
        self.repeat   = max(1, int(repeat_factor))
        self.max_shift= int(max_shift_bins)

        assert len(self.low_np) == len(self.high_np) == len(self.coords), "Mismatched dataset lengths"
        assert self.low_np.shape[1:] == self.high_np.shape[1:], "Low/High shapes must match (same resolution)"

    def __len__(self):
        return len(self.coords) * self.repeat

    def __getitem__(self, idx):
        true_idx = idx % len(self.coords)
        lo  = torch.tensor(self.low_np[true_idx],  dtype=torch.float32)  # (1,512,512)
        hi  = torch.tensor(self.high_np[true_idx], dtype=torch.float32)  # (1,512,512)
        c   = tuple(self.coords[true_idx])

        if self.augment:
            if torch.rand(1) < 0.5:
                lo = transpose_diag(lo); hi = transpose_diag(hi)
            if torch.rand(1) < 0.5:
                lo = flip_both(lo); hi = flip_both(hi)
            if (self.max_shift > 0) and (torch.rand(1) < 0.5):
                k = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
                if k != 0:
                    lo = roll_same(lo, k); hi = roll_same(hi, k)
            if torch.rand(1) < 0.3:
                lo = intensity_jitter(lo); hi = intensity_jitter(hi)

        # keep inputs in the VAE’s expected range
        lo = lo.clamp_(0.0, 1.0)
        hi = hi.clamp_(0.0, 1.0)

        return lo, hi, c


def custom_collate(batch):
    lows, highs, coords = zip(*batch)
    coords = ensure_coord_tuples(coords)
    return torch.stack(lows, 0), torch.stack(highs, 0), coords


# =============================== config =======================================

class TrainingConfig:
    def __init__(self):
        # -------- data paths (same resolution, e.g., 512×512 @ 25kb) ----------
        self.low_path   = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/ds25kb_lowdepth_dataset.npy"
        self.high_path  = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/ds25kb_highdepth_dataset.npy"
        self.coords_path= "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/ds25kb_hic_window_coords.npy"

        # hold-out chromosomes (prevent leakage)
        self.val_chroms  = ["chr8"]
        self.test_chroms = ["chr11"]

        # effective dataset expansion
        self.repeat_factor_train = 1
        self.max_shift_bins = 2

        # -------- single VAE checkpoint (512×512) -----------------------------
        self.vae_512_path = "/scratch/rnd-rojas/Manan/vq_highres_results_downsample_k4096/vae.best_pearson_corr.pt"

        # DNA conditioning (off; can enable when ready)
        self.use_dna = False
        self.dna_encoder_type = "simple"
        self.dna_embedding_dim = 256
        self.genome_fasta = "/scratch/rnd-rojas/Manan/hg19.fa"

        # ----------------- transformer arch -----------------------------------
        self.transformer_dim = 512
        self.transformer_depth = 8
        self.transformer_heads = 8
        self.transformer_dim_head = 64

        # ----------------- training ------------------------------------------
        self.batch_size = 2
        self.gradient_accumulation_steps = 8
        self.num_epochs = 100

        # optimizer
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.betas = (0.9, 0.95)
        self.warmup_steps = 1000

        # scheduling / logging
        self.use_mixed_precision = False
        self.log_every = 200
        self.eval_every = 2_000
        self.save_every = 1_000

        # mask schedule: anneal high → low
        self.mask_start = 0.60
        self.mask_end   = 0.10

        # misc
        self.val_fraction_backup = 0.10

        # mask visualization
        self.visualize_mask_every = 1000
        self.save_mask_visualizations = True

        # -------- VQGAN (informational; actual weights loaded from path) ------
        # Keep here for clarity if you need to reinit later
        self.vae_cfg = dict(
            dim=256, channels=1, layers=3,      # 512 -> 64x64 latent (4096 tokens)
            codebook_size=4096,
            lookup_free_quantization=False,
            vq_kwargs=dict(codebook_dim=256, commitment_weight=0.55, decay=0.99),
            l2_recon_loss=True, use_vgg_and_gan=False
        )


# =============================== trainer ======================================

class MuseTrainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.writer = SummaryWriter('runs/muse_training_singlevae')
        wandb.init(project="muse-hic", config=vars(cfg))

        self.current_epoch = 0
        self.global_step   = 0
        self.best_val_loss = float("inf")

        self._load_data()
        self._init_models()
        self._init_optim()


        # step budget (for LR + mask anneal)
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / max(1, self.cfg.gradient_accumulation_steps)))
        self.total_steps = self.cfg.num_epochs * steps_per_epoch

    # ---------------- data ----------------
    def _load_data(self):
        print("Loading arrays ...")
        lo = np.load(self.cfg.low_path,  mmap_mode="r")   # (N,1,512,512) low-depth
        hi = np.load(self.cfg.high_path, mmap_mode="r")   # (N,1,512,512) high-depth
        coords = np.load(self.cfg.coords_path, allow_pickle=True)

        coords = [tuple(c) for c in coords.tolist()]
        print(f"N={len(coords)} | lo {lo.shape} | hi {hi.shape}")

        if lo.shape != hi.shape:
            print(f"[WARN] Low/High shapes differ: {lo.shape} vs {hi.shape}. This pipeline expects SAME shapes.")

        # split by chromosomes (strict)
        chroms = [c[0] for c in coords]
        idx_val  = [i for i, ch in enumerate(chroms) if ch in self.cfg.val_chroms]
        idx_test = [i for i, ch in enumerate(chroms) if ch in self.cfg.test_chroms]
        idx_train= [i for i, ch in enumerate(chroms) if (ch not in self.cfg.val_chroms) and (ch not in self.cfg.test_chroms)]

        if len(idx_val) == 0:  # backup: random val split if specified chr not present
            n = len(coords)
            nval = max(1, int(self.cfg.val_fraction_backup * n))
            idx = np.random.permutation(n)
            idx_val = idx[:nval].tolist()
            idx_train = idx[nval:].tolist()
            print(f"[WARN] val_chroms not present; using random {len(idx_val)} val windows")

        lo_tr = lo[idx_train]; hi_tr = hi[idx_train]; coords_tr = [coords[i] for i in idx_train]
        lo_va = lo[idx_val];   hi_va = hi[idx_val];   coords_va = [coords[i] for i in idx_val]

        self.train_ds = HiCDataset(lo_tr, hi_tr, coords_tr,
                                   augment=True,
                                   repeat_factor=self.cfg.repeat_factor_train,
                                   max_shift_bins=self.cfg.max_shift_bins)
        self.val_ds   = HiCDataset(lo_va, hi_va, coords_va,
                                   augment=False,
                                   repeat_factor=1,
                                   max_shift_bins=0)

        self.train_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size,
                                       shuffle=True, num_workers=8, pin_memory=True,
                                       collate_fn=custom_collate, drop_last=True,
                                       persistent_workers=True)
        self.val_loader   = DataLoader(self.val_ds, batch_size=self.cfg.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True,
                                       collate_fn=custom_collate,
                                       persistent_workers=True)

        print(f"Train windows (unique): {len(coords_tr)}  | repeat x{self.cfg.repeat_factor_train}  -> ~{len(self.train_ds)}")
        print(f"Val windows (held-out chroms): {len(coords_va)}")

    @torch.no_grad()
    def _log_vae_recon(self, x, step, name="HR"):
        """
        Encode/decode a batch, print stats, and save before/after panel.
        x: [B,1,H,W] already on device and in the SAME numeric range used for training.
        """
        self.vae.eval()
        fmap, _, _ = self.vae.encode(x)
        rec = self.vae.decode(fmap)

        mse = torch.mean((rec - x)**2).item()
        rng = float((x.max() - x.min()).clamp_min(1e-6).item())
        ps  = 10.0 * math.log10((rng ** 2) / (mse + 1e-12))

        print(f"[VAE {name}] step={step}  mse={mse:.4f}  psnr≈{ps:.2f} dB  "
            f"in:[{x.min().item():.3f},{x.max().item():.3f}]  "
            f"rec:[{rec.min().item():.3f},{rec.max().item():.3f}]")

        panel_path = save_vae_recon_panel(x[:4], rec[:4], step, name)
        try:
            wandb.log({f"vae/{name}_mse": mse,
                    f"vae/{name}_psnr": ps,
                    f"vae/{name}_panel": wandb.Image(panel_path)},
                    step=step)
        except Exception:
            pass

    # ---------------- models ----------------
    def _infer_seq_len(self, vae, image_size=512):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size, device=self.device)
            fmap, _, _ = vae.encode(dummy)   # (1, f, f)
            f = fmap.shape[-1]
        return f * f

    def _init_models(self):
        print("Loading single VAE (512×512) ...")
        # Init a VAE with the expected architecture; then load weights
        self.vae = VQGanVAE(**self.cfg.vae_cfg).to(self.device)
        print(f"Loading VAE weights from: {self.cfg.vae_512_path}")
        self.vae.load(self.cfg.vae_512_path)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        # DNA encoder (optional)
        self.dna_encoder = None
        if self.cfg.use_dna:
            try:
                self.dna_encoder = create_dna_encoder(
                    encoder_type=self.cfg.dna_encoder_type,
                    genome_fasta=self.cfg.genome_fasta,
                    embedding_dim=self.cfg.dna_embedding_dim
                ).to(self.device)
                print(f"DNA encoder: {self.cfg.dna_encoder_type}")
            except Exception as e:
                print(f"[WARN] DNA encoder failed to init: {e}")
                self.dna_encoder = None

        # Build transformer with seq_len inferred from VAE latent fmap
        seq_len = self._infer_seq_len(self.vae, image_size=512)
        print(f"[Transformer] inferred seq_len = {seq_len}")

        self.transformer = MaskGitTransformer(
            num_tokens=self.vae.codebook_size,
            seq_len=seq_len,
            dim=self.cfg.transformer_dim,
            depth=self.cfg.transformer_depth,
            heads=self.cfg.transformer_heads,
            dim_head=self.cfg.transformer_dim_head,
            dna_encoder=self.dna_encoder,
            self_cond=True
        ).to(self.device)

        # Single MaskGit: generate high-depth from low-depth, same size conditioning
        self.maskgit = MaskGit(
            vae=self.vae,
            transformer=self.transformer,
            image_size=512,
            cond_vae=self.vae,        # same VAE for cond
            cond_image_size=512,      # SAME spatial size
            cond_drop_prob=0.0,       # <— do NOT drop LR during training (for now)
            self_cond_prob=0.5,       # <— reduce a bit; lets LR drive more
            no_mask_token_prob=0.10    # <— always use mask token; no “keep-as-is”
        ).to(self.device)

            # after creating self.vae and self.maskgit
        for m in (self.vae, self.maskgit.vae, self.maskgit.cond_vae):
            if m is not None:
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

            # --- put the guards HERE ---
        assert not any(p.requires_grad for p in self.maskgit.vae.parameters()), "VAE should be frozen"
        assert not any(p.requires_grad for p in self.maskgit.cond_vae.parameters()), "cond VAE should be frozen"

        # initial mask schedule (will be annealed per-step)
        self._set_mask_ratio(self.cfg.mask_start)

        # quick health check of VAE on a tiny batch
        self._check_vae_health()

        # --- grab a tiny batch for VAE sanity (prefer val, fallback to train)
        try:
            lo_val, hi_val, _ = next(iter(self.val_loader))
        except StopIteration:
            lo_val, hi_val, _ = next(iter(self.train_loader))

        # put on device, ensure correct range for your VAE ([0,1]), keep a few examples
        hi_val = hi_val.to(self.device).float().clamp(0.0, 1.0)[:4]

        # save before/after panel + log metrics to W&B
        self._log_vae_recon(hi_val, step=self.global_step, name="HR_init")

        # also print a small recon metric to stdout (same batch as above)
        with torch.no_grad():
            fmap, _, _ = self.vae.encode(hi_val)
            rec = self.vae.decode(fmap)
            mse = torch.mean((rec - hi_val)**2).item()
            rng = float((hi_val.max() - hi_val.min()).clamp_min(1e-6).item())
            ps  = 10.0 * math.log10((rng**2) / (mse + 1e-12))
        print(f"[VAE] real-batch recon MSE: {mse:.4f}  PSNR: {ps:.2f} dB")
        try:
            wandb.log({"vae/real_batch_mse": mse, "vae/real_batch_psnr": ps}, step=self.global_step)
        except Exception:
            pass

    def _check_vae_health(self):
        self.vae.to(self.device).eval()
        with torch.no_grad():
            x = torch.randn(2, 1, 512, 512, device=self.device)
            rec = self.vae(x)
            print(f"[VAE] device: {next(self.vae.parameters()).device}, rec shape: {rec.shape}, "
                  f"range: [{rec.min().item():.3f}, {rec.max().item():.3f}]")
            mse = torch.mean((rec - x)**2).item()
            print(f"[VAE] quick MSE: {mse:.4f}")

    # ---------------- optim / sched ----------------
    def _init_optim(self):
        params = list(self.transformer.parameters())
        if getattr(self.maskgit, 'token_critic', None) is not None:
            params += list(self.maskgit.token_critic.parameters())
        self.opt = AdamW(params, lr=self.cfg.learning_rate,
                 betas=self.cfg.betas, weight_decay=self.cfg.weight_decay)
        self.scaler = None  # not using AMP by default

    def _set_mask_ratio(self, max_ratio):
        def sched(t):  # cosine down to 0, scaled to max_ratio
            return max_ratio * torch.cos(t * math.pi * 0.5)
        self.maskgit.noise_schedule = sched

    # ---------------- training/eval ----------------
    def train(self):
        print("Starting training (single VAE, same-size conditioning) ...")
        start_time = time.time()

        for epoch in range(self.current_epoch, self.cfg.num_epochs):
            self.current_epoch = epoch
            avg_loss = self._train_epoch()
            print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")
            if (epoch + 1) % max(1, self.cfg.save_every // max(1, len(self.train_loader))) == 0:
                self._save_checkpoint(f"_epoch_{epoch}")

        print(f"Done in {time.time() - start_time:.1f}s")
        self.writer.close(); wandb.finish()

    def _train_epoch(self):
        self.maskgit.train()
        # keep VAEs frozen / in eval so codebooks don’t move
        if self.maskgit.vae is not None:       self.maskgit.vae.eval()
        if self.maskgit.cond_vae is not None:  self.maskgit.cond_vae.eval()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for bidx, (lo, hi, coords) in enumerate(pbar):
            coords = ensure_coord_tuples(coords)
            lo = lo.to(self.device).float()   # low-depth input (cond)
            hi = hi.to(self.device).float()   # high-depth target

            # ---- LR + mask schedule anneal (per-step)
            progress = min(1.0, self.global_step / max(1, self.total_steps))
            lr_mult = warmup_cosine(self.global_step, self.cfg.warmup_steps, self.total_steps, min_lr=0.1)
            for g in self.opt.param_groups:
                g['lr'] = self.cfg.learning_rate * lr_mult
            max_mask = self.cfg.mask_start + (self.cfg.mask_end - self.cfg.mask_start) * progress
            self._set_mask_ratio(max_mask)

            if bidx % self.cfg.gradient_accumulation_steps == 0:
                self.opt.zero_grad(set_to_none=True)

            # Forward (no autocast unless you enable AMP)
            loss = self.maskgit(
                hi,
                dna_coords=coords if self.cfg.use_dna else None,
                cond_images=lo,
                train_only_generator=False
            )
            loss = loss / self.cfg.gradient_accumulation_steps

            # mask visualization (existing)
            if self.cfg.visualize_mask_every > 0 and (self.global_step % self.cfg.visualize_mask_every == 0):
                try:
                    visualize_mask_once(self.maskgit, hi, self.global_step, save_dir="mask_visualizations")
                    tprobe = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999], device=hi.device)
                    rprobe = self.maskgit.noise_schedule(tprobe)
                    print(f"[mask-sched] t={tprobe.tolist()} -> ratio={rprobe.tolist()}")
                    # >>> NEW: VAE before/after panels for both targets and condition
                    self._log_vae_recon(hi[:4], step=self.global_step, name="HR_train")
                    self._log_vae_recon(lo[:4], step=self.global_step, name="LR_cond")
                except Exception as e:
                    print(f"viz failed: {e}")

            if not torch.isfinite(loss):
                print(f"⚠️  Non-finite loss detected: {loss.item()}, skipping batch")
                continue

            loss.backward()

            if (bidx + 1) % self.cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.maskgit.parameters(), 1.0)
                self.opt.step()

            epoch_losses.append(loss.item() * self.cfg.gradient_accumulation_steps)
            self.global_step += 1

            # logging
            if self.global_step % self.cfg.log_every == 0:
                avg = float(np.mean(epoch_losses[-self.cfg.log_every:]))
                pbar.set_postfix(loss=f"{avg:.4f}")
                self.writer.add_scalar("train/loss", avg, self.global_step)
                wandb.log({"train_loss": avg, "mask_max": max_mask, "step": self.global_step})

            # eval
            if self.global_step % self.cfg.eval_every == 0:
                v = self._validate()
                self.writer.add_scalar("val/loss", v, self.global_step)
                wandb.log({"val_loss": v, "step": self.global_step})
                if v < self.best_val_loss:
                    self.best_val_loss = v
                    self._save_checkpoint("best")

            if bidx % 50 == 0:
                torch.cuda.empty_cache()

        return float(np.mean(epoch_losses))

    @torch.no_grad()
    def _validate(self):
        self.maskgit.eval()
        losses = []
        for lo, hi, coords in self.val_loader:
            coords = ensure_coord_tuples(coords)
            lo = lo.to(self.device).float()
            hi = hi.to(self.device).float()
            l = self.maskgit(
                hi,
                dna_coords=coords if self.cfg.use_dna else None,
                cond_images=lo,
                train_only_generator=True
            )
            losses.append(l.item())

        # VAE codebook health on a small batch
        try:
            pl, ul = codebook_perplexity(self.vae, hi)
            wandb.log({"val/vae_perplexity": pl, "val/vae_used_codes": ul})
        except Exception:
            pass

        self.maskgit.train()
        return float(np.mean(losses))

    def _save_checkpoint(self, suffix=""):
        pkg = dict(
            epoch=self.current_epoch,
            step=self.global_step,
            best_val=self.best_val_loss,
            maskgit=self.maskgit.state_dict(),
            opt=self.opt.state_dict(),
            cfg=self.cfg.__dict__
        )
        path = f"checkpoint{suffix}.pt"
        torch.save(pkg, path)
        print(f"Saved {path}")


# =============================== main =========================================

if __name__ == "__main__":
    cfg = TrainingConfig()
    trainer = MuseTrainer(cfg)
    trainer.train()
