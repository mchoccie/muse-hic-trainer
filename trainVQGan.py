import torch
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer
import torch

# Load the saved model state
state_dict = torch.load('/scratch/rnd-rojas/Manan/baseResultsHighresolution/vae.49000.pt', map_location='cpu')  # or 'cuda' if needed

# View top-level keys (usually just 'state_dict' or the model's layer names)
print(state_dict.keys())
# vae = VQGanVAE(
#     dim = 256,
#     codebook_size = 1024,
#     lookup_free_quantization=False,
#     l2_recon_loss=True,
#     vq_kwargs=dict(commitment_weight=1.5, decay=0.99),
#     use_vgg_and_gan = False
# )

# # 256 * 256 at 50000 Resolution -- Low resolution
# # 512 * 512 at 25000 Resolution - High Resolution

# # train on folder of images, as many images as possible
# print(vae)
# trainer = VQGanVAETrainer(
#     vae = vae,
#     image_size = 256,             # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
#     folder = '/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_dataset_50kb.npy',
#     batch_size = 4,
#     grad_accum_every = 8,
#     num_train_steps = 50000,
#     results_folder = './qv_results'
    
# ).cuda()

# trainer.train()