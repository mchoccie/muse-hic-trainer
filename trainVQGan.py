import torch
# Use the Hi-C weighted VQGanVAE
#from vqgan_vae_hic_weighted import VQGanVAE  # <-- changed import
from muse_maskgit_pytorch.vqgan_vae_hic_weighted import VQGanVAE
from muse_maskgit_pytorch import VQGanVAETrainer

# Load the saved model state
# state_dict = torch.load('/scratch/rnd-rojas/Manan/baseResultsHighresolution/vae.49000.pt', map_location='cpu')  # or 'cuda' if needed

# # View top-level keys (usually just 'state_dict' or the model's layer names)
# print(state_dict.keys())

# Instantiate the Hi-C weighted VQGanVAE
vae = VQGanVAE(
    dim = 256,
    codebook_size = 1024,
    lookup_free_quantization=False,
    l2_recon_loss=True,
    vq_kwargs=dict(commitment_weight=1.5, decay=0.99),
    use_vgg_and_gan = False
)
#vae.load_state_dict(state_dict)

# Example: If you want to test the forward pass with weighted loss
# dummy_img = torch.randn(2, 1, 256, 256)
# loss = vae(dummy_img, return_loss=True, use_hic_weighted_loss=True, hic_weight_alpha=1.0)
# print('Weighted Hi-C loss:', loss)

# # 256 * 256 at 50000 Resolution -- Low resolution
# # 512 * 512 at 25000 Resolution - High Resolution

# # train on folder of images, as many images as possible
# print(vae)
trainer = VQGanVAETrainer(
    vae = vae,
    image_size = 256,
    folder = '/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/lowres_dataset.npy',
    batch_size = 4,
    grad_accum_every = 8,
    num_train_steps = 50000,
    results_folder = './qv_results2'
    # Note: If VQGanVAETrainer does not support passing use_hic_weighted_loss, you may need to modify it
)

trainer.train()