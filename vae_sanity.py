import torch, numpy as np
from vqgan_vae_hic_weighted import VQGanVAE

ckpt = "/scratch/rnd-rojas/Manan/vq_highres_results_gpt5/vae.best_srcc.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VQGanVAE(
    dim=384, channels=1, layers=4,
    codebook_size=2048,
    lookup_free_quantization=False,          # DISCRETE VQ (must match training)
    vq_kwargs=dict(commitment_weight=0.4, decay=0.99),
    use_vgg_and_gan=False
).to(device).eval()

sd = torch.load(ckpt, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]

missing, unexpected = vae.load_state_dict(sd, strict=False)
print("[VAE load] missing:", missing)
print("[VAE load] unexpected:", unexpected)

# --- Inspect any 2D tensors that look like a codebook
cb_shapes = [(k, tuple(v.shape)) for k,v in vae.state_dict().items() if getattr(v, "ndim", 0) == 2]
print("2D params:", cb_shapes)

# Quick encode/decode + utilization check
arr = np.load("/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/highres_dataset_gpt5.npy", mmap_mode="r")
x = torch.from_numpy(arr[:4]).float().to(device)                  # (4,1,512,512)
with torch.no_grad():
    fmap, idx, _ = vae.encode(x)                                  # idx: (4, 32, 32)
    y = vae.decode(fmap)

u = torch.unique(idx)
print("Unique codes in batch:", u.numel(), "min/max idx:", idx.min().item(), idx.max().item())
print("MAE recon:", torch.mean(torch.abs(x - y)).item())