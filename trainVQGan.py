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

# LOWRES_NPY   = '/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/lowres_dataset_gpt5.npy'  # (N,1,256,256)
# RESULTS_DIR  = './vq_lowres_results_gpt5'
# WANDB_PROJECT = 'vqgan-hic-training'
# WANDB_RUN     = 'vqgan-lowres-gpt5'

# # ---- VAE for LOW-RES tiles (256x256) ----
# # vae = VQGanVAE(
# #     dim = 256,                 # low-res encoder width (matches your previous setup)
# #     channels = 1,
# #     layers = 4,                # 256 -> 16x16 latent grid
# #     codebook_size = 1024,      # vocab size for 16x16 tokens
# #     lookup_free_quantization = False,  # must match how you trained the checkpoint
# #     vq_kwargs = dict(
# #         codebook_dim      = 256,
# #         commitment_weight = 0.4,
# #         decay             = 0.99
# #     ),
# #     # Reconstruction: L1 (no perceptual/GAN for Hi-C)
# #     l2_recon_loss = False,
# #     use_vgg_and_gan = False
# # )

# # trainer = VQGanVAETrainer(
# #     vae = vae,
# #     image_size = 256,                       # LOW-RES stage
# #     folder = LOWRES_NPY,                    # path to (N,1,256,256) NPY
# #     batch_size = 16,                        # adjust per GPU memory
# #     grad_accum_every = 4,                   # effective batch 16*4=64
# #     num_train_steps = 20_000,               # typical for low-res

# #     results_folder = RESULTS_DIR,

# #     # Plain L1 is recommended for Hi-C (no diagonal weighting)
# #     use_hic_weighted_loss = False,
# #     hic_weight_alpha = 0.7,                 # ignored when use_hic_weighted_loss=False

# #     codebook_log_interval = 200,
# #     wandb_project = WANDB_PROJECT,
# #     wandb_run_name = WANDB_RUN
# # )

# ---------------- config ----------------
# ---------------- config ----------------
HIGHRES_NPY   = '/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/ds25kb_lowdepth_dataset.npy'
RESULTS_DIR   = './vq_highres_results_downsample_detailedPreprocessing_lowres'
WANDB_PROJECT = 'vqgan-hic-training'
WANDB_RUN     = 'vqgan-highres-downsample-k4096'

# from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer

vae = VQGanVAE(
    dim = 256,
    channels = 1,
    layers = 3,                  # 512 -> 64x64 latent (4096 tokens)
    codebook_size = 4096,        # solid, stable size for 512px Hi-C
    lookup_free_quantization = False,   # EMA codebook

    vq_kwargs = dict(
        codebook_dim      = 256,
        # Slightly lower commitment to encourage healthier codebook usage
        # (reduces over-smoothing and dead codes vs 0.55)
        commitment_weight = 0.40,
        # Slower EMA is generally more stable at 512px + k=4096
        decay             = 0.995
    ),

    # Keep L2 per your note (Huber/L1 is also fine if your fork supports it)
    l2_recon_loss = True,
    use_vgg_and_gan = False,
)

trainer = VQGanVAETrainer(
    vae = vae,
    image_size = 512,
    folder = HIGHRES_NPY,

    batch_size = 4,
    grad_accum_every = 8,
    num_train_steps = 60_000,     # keep your longer run

    results_folder = RESULTS_DIR,

    # Hi-C weighted loss
    # 0.5–0.6 is a good starting band; 0.65 can over-focus diagonal early
    use_hic_weighted_loss = True,
    hic_weight_alpha = 0.55,

    codebook_log_interval = 200,
    wandb_project = WANDB_PROJECT,
    wandb_run_name = WANDB_RUN
)

trainer.train()


