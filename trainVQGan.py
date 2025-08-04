import torch
# Use the Hi-C weighted VQGanVAE
#from vqgan_vae_hic_weighted import VQGanVAE  # <-- changed import
from muse_maskgit_pytorch.vqgan_vae_hic_weighted import VQGanVAE
from muse_maskgit_pytorch import VQGanVAETrainer

# Instantiate the Hi-C weighted VQGanVAE
vae = VQGanVAE(
    dim              = 512,         # keeps encoder capacity
    codebook_size    = 4096,        # more tokens for hi‑res detail
    lookup_free_quantization = False,
    vq_kwargs = dict(
        codebook_dim       = 512,   # wider than 256 but still tractable
        commitment_weight  = 1.5,
        decay              = 0.99
    ),
    l2_recon_loss    = True,
    use_vgg_and_gan  = False
)

# Set up the trainer with Hi-C weighted loss and wandb logging
trainer = VQGanVAETrainer(
    vae = vae,
    image_size = 512,
    folder = '/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/highres_dataset.npy',
    batch_size = 4,
    grad_accum_every = 8,
    num_train_steps = 20000,
    results_folder = './qv_results_highres_new',  # Changed to completely new folder
    use_hic_weighted_loss = True,         # Enable Hi-C weighted loss
    hic_weight_alpha = 1.0,               # Set the alpha for weighting
    wandb_project = "hic-vqgan",         # wandb project name
    wandb_run_name = "hic_weighted_vqgan_run", # wandb run name
    codebook_log_interval = 200           # Log codebook stats every 200 steps
)

trainer.train()