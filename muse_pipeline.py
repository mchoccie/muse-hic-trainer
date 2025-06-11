import torch
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
from muse_maskgit_pytorch.dna_encoder import EnformerEncoder
import numpy as np

# ------------------------------------------------------------------
# 1)  load the files
# ------------------------------------------------------------------
hic_path   = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_dataset_50kb.npy"
coord_path = "/scratch/rnd-rojas/Manan/hic_window_coords.npy"

# Hi-C maps       – shape should be [N, 1, 256, 256]
hic_np = np.load(hic_path, mmap_mode="r")         # mmap saves RAM, remove if you plan to edit
print("Hi-C numpy  shape :", hic_np.shape, hic_np.dtype)

# window coordinates – saved as a pickled object array of tuples
coords_np = np.load(coord_path, allow_pickle=True)
coords = [tuple(c) for c in coords_np.tolist()]                 # convert to plain list
print("coord list len:", len(coords))
print("first coord   :", coords[0])               # e.g. ('chr1', 0, 12800000)

assert len(coords) == hic_np.shape[0], "mismatch in #windows"

# ------------------------------------------------------------------
# 2)  choose a batch (optional) and move to torch
# ------------------------------------------------------------------
batch_idx     = np.arange(4)          # first 4 windows for example
batch_images  = torch.from_numpy(hic_np[batch_idx]).float().cuda()  # [B, 1, 256, 256]
batch_coords  = [coords[i] for i in batch_idx]               # list of tuples

print("batch_images shape:", batch_images.shape)
print("batch_coords      :", batch_coords)
enformer_dir = '/scratch/rnd-rojas/Manan/enformer_local'
genome_fasta = '/scratch/rnd-rojas/Manan/hg19.fa'

dna_enc = EnformerEncoder(enformer_dir, genome_fasta)
vae = VQGanVAE(
    dim = 256,
    codebook_size = 1024,
    lookup_free_quantization=False,
    l2_recon_loss=True,
    vq_kwargs=dict(commitment_weight=1.5, decay=0.99),
    use_vgg_and_gan = False
).cuda()

vae.load('/scratch/rnd-rojas/Manan/qv_results/vae.49000.pt') # you will want to load the exponentially moving averaged VAE

transformer = MaskGitTransformer(
    num_tokens = 1024,   # codebook size
    dim        = 512,
    seq_len    = 1024,
    depth      = 8,
    dna_encoder = dna_enc,   # ← plug in here
)

transformer = transformer.cuda()  # 👈 Move it AFTER initialization

maskgit = MaskGit(
    vae           = vae,
    transformer   = transformer,
    image_size    = 256,
    cond_image_size = None,  # not doing SR here
)

# /scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_dataset_50kb.npy -- these are the Hi-C maps
# /scratch/rnd-rojas/Manan/hic_window_coords.npy -- these are the coordinates of the Hi-C windows
loss = maskgit(
    batch_images,                         # [B, 1, 256, 256] Hi-C maps
    dna_coords = batch_coords,      # list[ (chrom,start,end) ] len==B
)