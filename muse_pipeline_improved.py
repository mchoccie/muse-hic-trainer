# Updated muse_pipeline_improved.py
import os, math, time
from pathlib import Path
from functools import partial

# Reduce torch.compile verbosity
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# ---- your packages ----
from muse_maskgit_pytorch import MaskGit, MaskGitTransformer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'muse-maskgit-pytorch'))
from vqgan_vae_hic_weighted import VQGanVAE
from muse_maskgit_pytorch.improved_dna_encoders import create_dna_encoder

# ============================================================
# utils
# ============================================================

def exists(x): return x is not None

def psnr(x, y, max_val=None):
    # x,y: (B,1,H,W), in z-score space is fine – use dynamic range if not given
    mse = F.mse_loss(x, y)
    if max_val is None:
        # 6 is a typical z-score clip for viewing; not important for monitoring
        max_val = 6.0
    return 10 * math.log10((max_val ** 2) / (mse.item() + 1e-12))

@torch.no_grad()
def codebook_perplexity(vq: VQGanVAE, batch):
    # batch: (B,1,H,W)
    _, ids, _ = vq.encode(batch)     # (B, h, w)
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
    Recreates MaskGit.forward() masking:
      - tokenizes if floats
      - uses maskgit.noise_schedule(t)
      - guarantees at least 1 token masked (clamp(min=1))
    Returns (ids_flat, mask_bool, f) where:
      ids_flat : (B, S) original token ids
      mask_bool: (B, S) boolean mask (True where masked)
      f        : latent fmap size (e.g. 64 for 512px)
    """
    # tokenize if needed
    if images_or_ids.dtype == torch.float:
        _, ids, _ = maskgit.vae.encode(images_or_ids)     # (B, f, f)
    else:
        ids = images_or_ids

    B, f, _ = ids.shape
    ids_flat = ids.view(B, -1)
    S = ids_flat.size(1)
    device = ids_flat.device

    # same randomness as forward()
    t = torch.rand((B,), device=device)
    mask_ratio = maskgit.noise_schedule(t)               # tensor in [0, max_ratio]
    # safety: if the schedule accidentally returns a python float, tensor-ize it
    if not torch.is_tensor(mask_ratio):
        mask_ratio = torch.tensor(mask_ratio, device=device).expand(B)
    k = (S * mask_ratio).round().clamp(min=1).long()     # at least 1 token masked

    # random positions
    perm = torch.rand((B, S), device=device).argsort(dim=-1)
    mask_bool = perm < k[:, None]                        # (B, S) True where masked
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

    # take first item for display
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

# ============================================================
# symmetry-safe spatial ops
# ============================================================

def flip_both(x):
    # flip horizontally and vertically to preserve main diagonal orientation
    return torch.flip(x, dims=[-2, -1])

def transpose_diag(x):
    # reflect across main diagonal (H,W) -> (W,H), but shapes are square
    return x.transpose(-2, -1)

def roll_same(x, k):
    # roll same shift along both axes; k can be negative
    return torch.roll(torch.roll(x, shifts=k, dims=-1), shifts=k, dims=-2)

def intensity_jitter(x, scale_range=(0.95, 1.05), bias_range=(-0.05, 0.05)):
    s = torch.empty(1, device=x.device).uniform_(*scale_range).item()
    b = torch.empty(1, device=x.device).uniform_(*bias_range).item()
    return x * s + b
    
def load_vae_from_ckpt(ckpt_path, device, default_dim, default_codebook_size):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # infer dim
    ckpt_dim = None
    for k in ("enc_dec.encoders.0.weight", "module.enc_dec.encoders.0.weight"):
        if k in sd:
            ckpt_dim = sd[k].shape[0]
            break

    # infer codebook size (best effort)
    ckpt_codebook = None
    for k, v in sd.items():
        if ("codebook" in k or "embed" in k) and hasattr(v, "ndim") and v.ndim == 2:
            ckpt_codebook = v.shape[0]
            break

    dim = ckpt_dim if ckpt_dim is not None else default_dim
    codebook_size = ckpt_codebook if ckpt_codebook is not None else default_codebook_size

    vae = VQGanVAE(
        dim=dim,
        codebook_size=codebook_size,
        lookup_free_quantization=True,
        use_vgg_and_gan=False
    )  # create on CPU first

    model_sd = vae.state_dict()
    compatible = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    vae.load_state_dict(compatible, strict=False)

    # 🔑 ensure weights are on the correct device
    vae = vae.to(device).float().eval()

    print(f"[VAE] loaded {ckpt_path}")
    print(f"      dim={dim}, codebook_size={codebook_size}, loaded_keys={len(compatible)}")
    return vae


# ============================================================
# Dataset with symmetry-preserving augmentation and repeat factor
# ============================================================

class HiCDataset(Dataset):
    """
    Expects arrays:
      - lowres: (N, 1, 256, 256)  z-scored
      - highres:(N, 1, 512, 512)  z-scored
      - coords : list[ (chrom, start_bp, end_bp) ]
    *augment* applies only symmetry-safe transforms.
    *repeat_factor* virtually multiplies dataset size by sampling augmentations K times.
    *max_shift_bins* does equal x/y 'roll' in bins (keeps diagonal sense).
    """
    def __init__(self, lowres_np, highres_np, coords, augment=True, repeat_factor=1, max_shift_bins=4):
        self.lowres_np  = lowres_np
        self.highres_np = highres_np
        self.coords     = coords
        self.augment    = augment
        self.repeat     = max(1, int(repeat_factor))
        self.max_shift  = int(max_shift_bins)

    def __len__(self):
        return len(self.coords) * self.repeat

    def __getitem__(self, idx):
        true_idx = idx % len(self.coords)

        # Use torch.tensor for faster conversion
        lo  = torch.tensor(self.lowres_np[true_idx], dtype=torch.float32)   # (1,256,256)
        hi  = torch.tensor(self.highres_np[true_idx], dtype=torch.float32)  # (1,512,512)
        c   = tuple(self.coords[true_idx])

        if self.augment:
            # 1) diagonal transpose with p=0.5
            if torch.rand(1) < 0.5:
                lo = transpose_diag(lo)
                hi = transpose_diag(hi)

            # 2) flip both axes with p=0.5
            if torch.rand(1) < 0.5:
                lo = flip_both(lo)
                hi = flip_both(hi)

            # 3) equal x/y tile jitter (roll) with p=0.5
            if (self.max_shift > 0) and (torch.rand(1) < 0.5):
                k = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
                if k != 0:
                    lo = roll_same(lo, k)
                    hi = roll_same(hi, 2 * k)  # 512 is 2× finer; keep shift consistent in bins

            # 4) mild intensity jitter with p=0.3
            if torch.rand(1) < 0.3:
                lo = intensity_jitter(lo)
                hi = intensity_jitter(hi)

        return lo, hi, c

def custom_collate(batch):
    lows, highs, coords = zip(*batch)
    coords = ensure_coord_tuples(coords)
    return torch.stack(lows, 0), torch.stack(highs, 0), coords

# ============================================================
# Config
# ============================================================

class TrainingConfig:
    def __init__(self):
        # ----------------- data paths -----------------
        self.hic_path_lowres = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/lowres_dataset_gpt5.npy"
        self.hic_path_highres = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/highres_dataset_gpt5.npy"
        self.coords_path = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_window_coords_gpt5.npy"

        # hold-out chromosomes (prevent leakage)
        self.val_chroms  = ["chr8"]
        self.test_chroms = ["chr11"]

        # effective dataset expansion
        self.repeat_factor_train = 1  # Reduced from 8 to speed up training
        self.max_shift_bins = 2  # Reduced from 4

        # pretrained VAE checkpoints
        self.vae_base_path    = "/scratch/rnd-rojas/Manan/vq_lowres_results_gpt5/vae.best_pearson_corr.pt"
        self.vae_highres_path = "/scratch/rnd-rojas/Manan/vq_highres_results_gpt5_layer_adjusted/vae.best_pearson_corr.pt"

        # DNA conditioning (off for now, can enable later)
        self.use_dna = False
        self.dna_encoder_type = "simple"
        self.dna_embedding_dim = 256
        self.genome_fasta = "/scratch/rnd-rojas/Manan/hg19.fa"

        # ----------------- transformer arch -----------------
        self.transformer_dim = 512
        self.transformer_depth = 8  # Back to original
        self.transformer_heads = 8
        self.transformer_dim_head = 64

        # ----------------- training -----------------
        self.batch_size = 2 # Back to optimized size
        self.gradient_accumulation_steps = 1  # Reduced from 2
        self.num_epochs = 100

        # optimizer
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.betas = (0.9, 0.95)
        self.warmup_steps = 5_000

        # scheduling / logging
        self.use_mixed_precision = False  # Using regular floats instead of mixed precision
        self.log_every = 200  # Reduced logging frequency
        self.eval_every = 2_000  # Reduced evaluation frequency
        self.save_every = 1_000

        # mask schedule: anneal high → low
        self.mask_start = 0.90
        self.mask_end   = 0.10

        # other
        self.critic_loss_weight = 0.5
        self.val_fraction_backup = 0.10
        
        # mask visualization
        self.visualize_mask_every = 1000  # Visualize masking every N steps (0 to disable)
        self.save_mask_visualizations = True  # Save mask visualizations to disk

        # ----------------- VQGAN configs -----------------

        # low-res VQGAN (stable, no collapse)
        self.vae_low_cfg = dict(
            dim=256,
            channels=1,
            layers=4,                # 256 → 16x16 latent
            codebook_size=1024,
            lookup_free_quantization=False,
            vq_kwargs=dict(
                codebook_dim=256,
                commitment_weight=0.4,   # stays lower → low-res was fine
                decay=0.99
            ),
            l2_recon_loss=False,
            use_vgg_and_gan=False
        )

        # high-res VQGAN (adjusted for stability)
        self.vae_high_cfg = dict(
            dim=256,                  # <= you trained with this
            channels=1,
            layers=3,                 # 512 -> 64x64 latent  = 4096 tokens
            codebook_size=1024,       # <= you trained with this
            lookup_free_quantization=False,
            vq_kwargs=dict(
                codebook_dim=256,
                commitment_weight=0.65,
                decay=0.995
            ),
            l2_recon_loss=False,
            use_vgg_and_gan=False
        )

# ============================================================
# Trainer
# ============================================================

class MuseTrainer:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.writer = SummaryWriter('runs/muse_training')
        wandb.init(project="muse-hic", config=vars(cfg))

        self._load_data()
        self._init_models()
        self._init_optim()

        self.current_epoch = 0
        self.global_step   = 0
        self.best_val_loss = float("inf")

        # step budget (for LR + mask anneal)
        self.total_steps = self.cfg.num_epochs * math.ceil(len(self.train_loader) / max(1, self.cfg.gradient_accumulation_steps))

    # ---------------- data ----------------
    def _load_data(self):
        print("Loading arrays ...")
        lo = np.load(self.cfg.hic_path_lowres, mmap_mode="r")     # (N,1,256,256)
        hi = np.load(self.cfg.hic_path_highres, mmap_mode="r")    # (N,1,512,512)
        coords = np.load(self.cfg.coords_path, allow_pickle=True)

        coords = [tuple(c) for c in coords.tolist()]
        print(f"N={len(coords)} | lo {lo.shape} | hi {hi.shape}")

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

        # datasets (train with repeat_factor >> 1 for effective 100k+ samples)
        self.train_ds = HiCDataset(lo_tr, hi_tr, coords_tr,
                                   augment=True,
                                   repeat_factor=self.cfg.repeat_factor_train,
                                   max_shift_bins=self.cfg.max_shift_bins)
        self.val_ds   = HiCDataset(lo_va, hi_va, coords_va,
                                   augment=False,
                                   repeat_factor=1,
                                   max_shift_bins=0)

        self.train_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size,
                                       shuffle=True, num_workers=12, pin_memory=True,
                                       collate_fn=custom_collate, drop_last=True,
                                       persistent_workers=True)
        self.val_loader   = DataLoader(self.val_ds, batch_size=self.cfg.batch_size,
                                       shuffle=False, num_workers=6, pin_memory=True,
                                       collate_fn=custom_collate,
                                       persistent_workers=True)

        print(f"Train windows (unique): {len(coords_tr)}  | repeat x{self.cfg.repeat_factor_train}  -> ~{len(self.train_ds)}")
        print(f"Val windows (held-out chroms): {len(coords_va)}")



    # ---------------- models ----------------
    def _init_models(self):
        print("Loading VAEs ...")
        
        # Create VAE instances with configs
        self.vae_base = VQGanVAE(**self.cfg.vae_low_cfg).to(self.device)
        self.vae_high = VQGanVAE(**self.cfg.vae_high_cfg).to(self.device)
        
        # Load pretrained weights
        print(f"Loading base VAE from: {self.cfg.vae_base_path}")
        self.vae_base.load(self.cfg.vae_base_path)
        print(f"Loading high-res VAE from: {self.cfg.vae_highres_path}")
        self.vae_high.load(self.cfg.vae_highres_path)
        
        # --- infer the true latent sequence lengths from the loaded VAEs ---
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
        # transformers - use actual VAE codebook sizes

        low_tokens  = 256
        high_tokens = 4096
        self.tr_low = MaskGitTransformer(
            num_tokens=self.vae_base.codebook_size, dim=self.cfg.transformer_dim, seq_len=low_tokens,
            depth=self.cfg.transformer_depth, heads=self.cfg.transformer_heads, dim_head=self.cfg.transformer_dim_head,
            dna_encoder=self.dna_encoder, self_cond=True
        ).to(self.device)

        self.tr_high = MaskGitTransformer(
            num_tokens=self.vae_high.codebook_size, dim=self.cfg.transformer_dim, seq_len=high_tokens,
            depth=self.cfg.transformer_depth, heads=self.cfg.transformer_heads, dim_head=self.cfg.transformer_dim_head,
            dna_encoder=self.dna_encoder, self_cond=True
        ).to(self.device)

        self.maskgit_low = MaskGit(
            vae=self.vae_base, transformer=self.tr_low, image_size=256,
            cond_drop_prob=0.1, self_cond_prob=0.9, no_mask_token_prob=0.1
        ).to(self.device)

        self.maskgit_high = MaskGit(
            vae=self.vae_high, transformer=self.tr_high, image_size=512, cond_vae=self.vae_base, 
            cond_image_size=256, cond_drop_prob=0.25, self_cond_prob=0.9, no_mask_token_prob=0.1
        ).to(self.device)

        # self._sync_transformer_seq_len(self.maskgit_low,  image_size=256, tag="LOW")
        # self._sync_transformer_seq_len(self.maskgit_high, image_size=512, tag="HIGH")

        # Compile models for faster training (PyTorch 2.0+) - DISABLED for cleaner output
        # try:
        #     self.maskgit_low = torch.compile(self.maskgit_low, mode="reduce-overhead")
        #     self.maskgit_high = torch.compile(self.maskgit_high, mode="reduce-overhead")
        #     print("✅ Models compiled with torch.compile")
        # except Exception as e:
        #     print(f"⚠️  torch.compile not available: {e}")
        print("⚠️  torch.compile disabled for cleaner output - enable later for speed")
        
        # Set models to eval mode for VAE health check
        self.vae_base.eval()
        self.vae_high.eval()

        # initial mask schedule (will be annealed per-step)
        self._set_mask_ratio(self.cfg.mask_start)

        # quick health check of VAEs on a tiny batch
        self._check_vaes()

        try:
            lo_val, hi_val, _ = next(iter(self.val_loader))   # one batch from val
        except StopIteration:
            lo_val, hi_val, _ = next(iter(self.train_loader)) # fallback to train

        hi_val = hi_val[:1].to(self.device).float()           # keep it tiny to save mem
        with torch.no_grad():
            fmap, _, _ = self.vae_high.encode(hi_val)
            hr_rec = self.vae_high.decode(fmap)
            mse = torch.mean((hr_rec - hi_val) ** 2).item()
            # optional: use your psnr() helper defined above
            psnr_val = psnr(hr_rec, hi_val)

        print(f"[VAE-high] real-batch recon MSE: {mse:.4f}  PSNR: {psnr_val:.2f} dB")
        try:
            wandb.log({"hr_vae/real_batch_mse": mse, "hr_vae/real_batch_psnr": psnr_val})
        except Exception:
            pass

    def _check_vaes(self):
        # Make absolutely sure models are on the right device
        self.vae_base.to(self.device).eval()
        self.vae_high.to(self.device).eval()

        with torch.no_grad():
            # Dummy grayscale images (B, C=1, H, W)
            lo = torch.randn(2, 1, 256, 256, device=self.device)
            hi = torch.randn(2, 1, 512, 512, device=self.device)

            # Option A: simple forward (returns recon)
            rec_lo = self.vae_base(lo)                      # shape (2,1,256,256)
            rec_hi = self.vae_high(hi)                      # shape (2,1,512,512)

            # Option B: explicit encode→decode (uncomment if you want to test the path)
            # fmap_lo, _, _ = self.vae_base.encode(lo)
            # rec_lo2 = self.vae_base.decode(fmap_lo)
            # fmap_hi, _, _ = self.vae_high.encode(hi)
            # rec_hi2 = self.vae_high.decode(fmap_hi)

            # Sanity prints
            first_weight = next(self.vae_base.parameters())
            print(f"[VAE-base] weight device: {first_weight.device}, input device: {lo.device}, rec device: {rec_lo.device}")
            print(f"[VAE-high] weight device: {next(self.vae_high.parameters()).device}, input device: {hi.device}, rec device: {rec_hi.device}")

            print(f"[VAE-base] recon shape: {rec_lo.shape}, dtype: {rec_lo.dtype}, range: [{rec_lo.min().item():.3f}, {rec_lo.max().item():.3f}]")
            print(f"[VAE-high] recon shape: {rec_hi.shape}, dtype: {rec_hi.dtype}, range: [{rec_hi.min().item():.3f}, {rec_hi.max().item():.3f}]")

            # quick MSE just to see it's numerically sane
            mse_lo = torch.mean((rec_lo - lo)**2).item()
            mse_hi = torch.mean((rec_hi - hi)**2).item()
            print(f"[VAE] quick MSE — lowres: {mse_lo:.4f}  highres: {mse_hi:.4f}")

    # ---------------- optim / sched ----------------
    def _init_optim(self):
        self.opt_low  = AdamW(self.maskgit_low.parameters(),  lr=self.cfg.learning_rate, betas=self.cfg.betas, weight_decay=self.cfg.weight_decay)
        self.opt_high = AdamW(self.maskgit_high.parameters(), lr=self.cfg.learning_rate, betas=self.cfg.betas, weight_decay=self.cfg.weight_decay)
        # No gradient scaler needed for regular float training
        self.scaler = None

    # dynamically set noise schedule to a scaled cosine with current max
    def _set_mask_ratio(self, max_ratio):
        # cosine(t) in [0,1] → [1→0]; we scale its amplitude to [0, max_ratio]
        def sched(t):
            # t is tensor in [0,1]; return scalar or tensor
            return max_ratio * torch.cos(t * math.pi * 0.5)
        self.maskgit_low.noise_schedule  = sched
        self.maskgit_high.noise_schedule = sched

    # ---------------- training/eval ----------------
    def train(self):
        print("Starting training ...")
        start_time = time.time()

        for epoch in range(self.current_epoch, self.cfg.num_epochs):
            self.current_epoch = epoch
            avg_loss = self._train_epoch()
            print(f"[Epoch {epoch}] train loss: {avg_loss:.4f}")
            self._save_checkpoint(f"_epoch_{epoch}")

        print(f"Done in {time.time() - start_time:.1f}s")
        self.writer.close(); wandb.finish()

    def _train_epoch(self):
        self.maskgit_low.train(); self.maskgit_high.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for bidx, (lo, hi, coords) in enumerate(pbar):
            coords = ensure_coord_tuples(coords)
            lo = lo.to(self.device); hi = hi.to(self.device)

            # ---- LR + mask schedule anneal (per-step)
            progress = min(1.0, self.global_step / max(1, self.total_steps))
            # lr mult
            lr_mult = warmup_cosine(self.global_step, self.cfg.warmup_steps, self.total_steps, min_lr=1e-2)
            for g in self.opt_low.param_groups:  g['lr']  = self.cfg.learning_rate * lr_mult
            for g in self.opt_high.param_groups: g['lr'] = self.cfg.learning_rate * lr_mult
            # mask max
            max_mask = self.cfg.mask_start + (self.cfg.mask_end - self.cfg.mask_start) * progress
            self._set_mask_ratio(max_mask)

            # ---- zero grads on accumulation boundaries
            if bidx % self.cfg.gradient_accumulation_steps == 0:
                self.opt_low.zero_grad(set_to_none=True)
                self.opt_high.zero_grad(set_to_none=True)

            # Ensure inputs are in float32
            lo = lo.float()
            hi = hi.float()
            
            # Forward pass without autocast
            #loss_lo = self.maskgit_low(lo, dna_coords=coords if self.cfg.use_dna else None, train_only_generator=False)
            loss_hi = self.maskgit_high(hi, dna_coords=coords if self.cfg.use_dna else None, cond_images=lo, train_only_generator=False)
            loss = (loss_hi) / self.cfg.gradient_accumulation_steps
            
            # Visualize masking periodically
            if self.cfg.visualize_mask_every > 0 and (self.global_step % self.cfg.visualize_mask_every == 0):
                try:
                    visualize_mask_once(self.maskgit_high, hi, self.global_step, save_dir="mask_visualizations")
                    # quick schedule sanity print
                    # sample schedule outputs at fixed times to see amplitude
                    tprobe = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.999], device=hi.device)
                    rprobe = self.maskgit_high.noise_schedule(tprobe)
                    print(f"[mask-sched] t={tprobe.tolist()} -> ratio={rprobe.tolist()}")
                except Exception as e:
                    print(f"mask viz failed: {e}")

            # Check for NaN/Inf before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  NaN/Inf loss detected: {loss.item()}, skipping batch")
                continue
            
            # Ensure loss is finite before backward pass
            if not torch.isfinite(loss):
                print(f"⚠️  Non-finite loss detected: {loss.item()}, skipping batch")
                continue
                
            loss.backward()

            if (bidx + 1) % self.cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.maskgit_low.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.maskgit_high.parameters(), 1.0)
                self.opt_low.step()
                self.opt_high.step()

            epoch_losses.append(loss.item() * self.cfg.gradient_accumulation_steps)
            self.global_step += 1

            # ---- logging
            if self.global_step % self.cfg.log_every == 0:
                avg = float(np.mean(epoch_losses[-self.cfg.log_every:]))
                lr_lo = self.opt_low.param_groups[0]['lr']; lr_hi = self.opt_high.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_lo:.2e}")
                self.writer.add_scalar("train/loss", avg, self.global_step)
                self.writer.add_scalar("lr/low", lr_lo, self.global_step)
                wandb.log({"train_loss": avg, "lr": lr_lo, "mask_max": max_mask, "step": self.global_step})

            # ---- eval
            if self.global_step % self.cfg.eval_every == 0:
                v = self._validate()
                self.writer.add_scalar("val/loss", v, self.global_step)
                wandb.log({"val_loss": v, "step": self.global_step})
                if v < self.best_val_loss:
                    self.best_val_loss = v
                    self._save_checkpoint("best")

            # memory hygiene - less frequent to reduce overhead
            if bidx % 50 == 0:
                torch.cuda.empty_cache()

        return float(np.mean(epoch_losses))

    @torch.no_grad()
    def _validate(self):
        self.maskgit_low.eval(); self.maskgit_high.eval()
        losses = []
        for lo, hi, coords in self.val_loader:
            coords = ensure_coord_tuples(coords)
            lo = lo.to(self.device); hi = hi.to(self.device)
            l1 = self.maskgit_low(lo, dna_coords=coords if self.cfg.use_dna else None, train_only_generator=True)
            l2 = self.maskgit_high(hi, dna_coords=coords if self.cfg.use_dna else None, cond_images=lo, train_only_generator=True)
            losses.append((l1 + l2).item())
        # also log quick VAE stats to detect regressions
        try:
            pl, ul = codebook_perplexity(self.vae_base, lo)
            ph, uh = codebook_perplexity(self.vae_high, hi)
            wandb.log({"val/vae_low_perp": pl, "val/vae_low_used": ul,
                       "val/vae_high_perp": ph, "val/vae_high_used": uh})
        except Exception:
            pass
        self.maskgit_low.train(); self.maskgit_high.train()
        return float(np.mean(losses))

    def _save_checkpoint(self, suffix=""):
        pkg = dict(
            epoch=self.current_epoch, step=self.global_step, best_val=self.best_val_loss,
            maskgit_low=self.maskgit_low.state_dict(),
            maskgit_high=self.maskgit_high.state_dict(),
            opt_low=self.opt_low.state_dict(), opt_high=self.opt_high.state_dict(),
            cfg=self.cfg.__dict__
        )
        path = f"checkpoint{suffix}.pt"
        torch.save(pkg, path)
        print(f"Saved {path}")

# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    cfg = TrainingConfig()
    trainer = MuseTrainer(cfg)
    trainer.train()
