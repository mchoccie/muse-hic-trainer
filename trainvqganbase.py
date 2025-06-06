import torch
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer
vae = VQGanVAE(
    dim = 256,
    codebook_size = 1024,
    use_vgg_and_gan = False
)

# train on folder of images, as many images as possible

trainer = VQGanVAETrainer(
    vae = vae,
    image_size = 128,             # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
    folder = '/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_dataset_25kb_highres.npy',
    batch_size = 4,
    grad_accum_every = 8,
    num_train_steps = 50000,
    results_folder = './baseResultsHighresolution',
).cuda()

trainer.train()