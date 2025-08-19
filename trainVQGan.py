# import torch
# # Use the Hi-C weighted VQGanVAE
# #from vqgan_vae_hic_weighted import VQGanVAE  # <-- changed import
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'muse-maskgit-pytorch'))
# from vqgan_vae_hic_weighted import VQGanVAE
# from muse_maskgit_pytorch import VQGanVAETrainer

# # ---- VAE for LOW-RES tiles (256x256) ----
# vae = VQGanVAE(
#     dim = 256,                 # encoder/decoder width; plenty for low-res
#     codebook_size = 1024,      # 512–1024 is the sweet spot; start with 1024
#     lookup_free_quantization = False,
#     vq_kwargs = dict(
#         codebook_dim      = 256,   # 256 is enough; 512 often overkill
#         commitment_weight = 0.4,   # 0.25–0.5 common; lower than your 1.5
#         decay             = 0.99   # EMA codebook update
#     ),
#     # Losses: L1 beats L2 for Hi-C; skip perceptual/GAN
#     l2_recon_loss = False,
#     use_vgg_and_gan = False,
#     # (if your class supports it)
#     ssim_loss = True,          # optional; if available in your fork
#     ssim_weight = 0.25         # optional
# )

# trainer = VQGanVAETrainer(
#     vae = vae,
#     image_size = 256,                         # LOW-RES stage!
#     folder = '/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/lowres_dataset_gpt5.npy',   # use your (N,1,256,256) NPY loader
#     batch_size = 8,                           # increase if GPU allows
#     grad_accum_every = 4,                     # raise effective batch size
#     num_train_steps = 20000,                  # 20k steps as requested
#     results_folder = './vq_lowres_results_gpt5',
#     use_hic_weighted_loss = True,             # Hi-C weighted loss enabled
#     hic_weight_alpha = 0.7,                   # tone down from 1.0 for stability
#     codebook_log_interval = 200,
#     wandb_project = 'vqgan-hic-training',     # wandb project name
#     wandb_run_name = 'vqgan-lowres-gpt5'      # wandb run name
# )

# trainer.train()



import os, sys
import torch

# make sure the repo is importable
sys.path.append(os.path.join(os.path.dirname(__file__), 'muse-maskgit-pytorch'))

# custom VQGAN (your weighted-loss variant)
from vqgan_vae_hic_weighted import VQGanVAE
# trainer from the package (ensure its type hint was changed to nn.Module as we discussed)
from muse_maskgit_pytorch import VQGanVAETrainer

# ---------------- config ----------------
HIGHRES_NPY = '/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/highres_dataset_gpt5.npy'  # shape (N,1,512,512)
RESULTS_DIR = './vq_highres_results_gpt5'
WANDB_PROJECT = 'vqgan-hic-training'
WANDB_RUN = 'vqgan-highres-gpt5'

# ---- VAE for HIGH-RES tiles (512x512) ----
vae = VQGanVAE(
    dim = 384,                 # 384–512 work; 384 is lighter on memory
    channels = 1,
    layers = 4,                # 512 -> 32x32 latent grid
    codebook_size = 2048,      # 1024–2048 typical for 512x512 Hi-C
    lookup_free_quantization = False,
    vq_kwargs = dict(
        codebook_dim      = 256,   # 256–384; 256 is usually enough
        commitment_weight = 0.4,   # 0.3–0.5 stable
        decay             = 0.99
    ),
    # Reconstruction: L1 (no perceptual/GAN for Hi-C)
    l2_recon_loss = False,
    use_vgg_and_gan = False
)

trainer = VQGanVAETrainer(
    vae = vae,
    image_size = 512,                         # HIGH-RES stage
    folder = HIGHRES_NPY,                     # (N,1,512,512) NPY
    batch_size = 4,                           # adjust per GPU
    grad_accum_every = 8,                     # raises effective batch
    num_train_steps = 50_000,                # train longer for hi-res
    results_folder = RESULTS_DIR,

    # optional weighted L1 (diagonal distance), safe to disable if you prefer plain L1
    use_hic_weighted_loss = True,
    hic_weight_alpha = 0.7,

    codebook_log_interval = 200,
    wandb_project = WANDB_PROJECT,
    wandb_run_name = WANDB_RUN
)


trainer.train()