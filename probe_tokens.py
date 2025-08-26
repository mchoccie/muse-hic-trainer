# probe_tokens.py
import os, sys, torch
sys.path.append(os.path.join(os.path.dirname(__file__), 'muse-maskgit-pytorch'))

from vqgan_vae_hic_weighted import VQGanVAE   # <- your VAE class

def probe_checkpoint(ckpt_path: str, image_size: int, vae_cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) build a VAE with the *training* config you used for that checkpoint
    vae = VQGanVAE(**vae_cfg).to(device).eval()

    # 2) load weights
    vae.load(ckpt_path)

    # 3) encode a dummy image to get the latent grid (h, w) and thus seq_len
    with torch.no_grad():
        dummy = torch.zeros(1, 1, image_size, image_size, device=device)
        _, ids, _ = vae.encode(dummy)          # ids: [1, h, w]
        h, w = ids.shape[-2:]
        seq_len = h * w
        K = vae.codebook_size

        # optional: quick range check (sample another random image)
        rnd = torch.randn_like(dummy)
        _, rnd_ids, _ = vae.encode(rnd)
        id_min, id_max = int(rnd_ids.min().item()), int(rnd_ids.max().item())

    name = os.path.basename(ckpt_path)
    print(f"\n{ name }")
    print(f"  image_size       : {image_size}")
    print(f"  latent grid      : {h} x {w}")
    print(f"  seq_len (tokens) : {seq_len}")
    print(f"  codebook_size    : {K}")
    print(f"  id range (sample): [{id_min}, {id_max}]  (should be within 0..{K-1})")

if __name__ == "__main__":
    # --- fill in your two VAEs' configs (these match the TrainingConfig you posted) ---
    low_cfg = dict(
        dim=256, channels=1, layers=4, codebook_size=1024,
        lookup_free_quantization=False,
        vq_kwargs=dict(codebook_dim=256, commitment_weight=0.4, decay=0.99),
        l2_recon_loss=False, use_vgg_and_gan=False
    )
    high_cfg = dict(
        dim=384, channels=1, layers=4, codebook_size=2048,
        lookup_free_quantization=False,
        vq_kwargs=dict(codebook_dim=256, commitment_weight=0.65, decay=0.995),
        l2_recon_loss=False, use_vgg_and_gan=False
    )

    # --- your paths ---
    low_path  = "/scratch/rnd-rojas/Manan/vq_lowres_results_gpt5/vae.best_pearson_corr.pt"
    high_path = "/scratch/rnd-rojas/Manan/vq_highres_results_gpt5/vae.best_pearson_corr.pt"

    probe_checkpoint(low_path,  image_size=256, vae_cfg=low_cfg)
    probe_checkpoint(high_path, image_size=512, vae_cfg=high_cfg)