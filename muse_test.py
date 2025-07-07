import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
from muse_maskgit_pytorch.dna_encoder import EnformerEncoder

# ---------------------------------------------------------------
# Load Hi-C Data (assuming .npy files)
# ---------------------------------------------------------------
lowres_np_path  = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/lowres_dataset.npy"
highres_np_path = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/highres_dataset.npy"  # optional
coords_path     = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_window_coords.npy"

lowres_np = np.load(lowres_np_path, mmap_mode="r")
coords = np.load(coords_path, allow_pickle=True)
coords = [tuple(c) for c in coords.tolist()]
assert len(coords) == lowres_np.shape[0], "Mismatch in coord count and images"

# ---------------------------------------------------------------
# Define Dataset
# ---------------------------------------------------------------
class HiCDataset(Dataset):
    def __init__(self, lowres_np, coords):
        self.lowres_np = lowres_np
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        lowres = torch.from_numpy(self.lowres_np[idx]).float()  # shape: [1, 256, 256]
        coord = self.coords[idx]  # tuple
        return lowres, coord

def collate_fn(batch):
    lows, coords = zip(*batch)
    return torch.stack(lows), list(coords)

dataset = HiCDataset(lowres_np, coords)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# ---------------------------------------------------------------
# Load Enformer DNA Encoder
# ---------------------------------------------------------------
enformer_dir = '/scratch/rnd-rojas/Manan/enformer_local'
genome_fasta = '/scratch/rnd-rojas/Manan/hg19.fa'
dna_encoder = EnformerEncoder(enformer_dir, genome_fasta)

# ---------------------------------------------------------------
# Load Pretrained VQGAN and MaskGit
# ---------------------------------------------------------------
vaeHighres = VQGanVAE(
    dim=256,
    codebook_size=1024,
    use_vgg_and_gan=False
).cuda()

vaeHighres.load('/scratch/rnd-rojas/Manan/baseResultsHighresolution/vae.49000.pt')

transformerLowRes = MaskGitTransformer(
    num_tokens = 1024,   # codebook size
    dim        = 512,
    seq_len    = 1024,
    depth      = 8,
    dna_encoder = dna_encoder,   # ← plug in here
)

transformerHighRes = MaskGitTransformer(
    num_tokens = 1024,   # codebook size
    dim        = 512,
    seq_len    = 1024,
    depth      = 8,
    dna_encoder = dna_encoder,   # ← plug in here
)


superres_maskgit = MaskGit(
    vae=vaeHighres,
    transformer=transformerHighRes,
    cond_image_size=256,
    image_size=512,
    cond_drop_prob=0.0
).cuda()

# ---------------------------------------------------------------
# Inference: Generate high-res Hi-C maps from low-res + coords
# ---------------------------------------------------------------
checkpoint_path = "/scratch/rnd-rojas/Manan/maskgit_highres.pt"
state_dict = torch.load(checkpoint_path)
superres_maskgit.load_state_dict(state_dict)
lowres_batch, coords_batch = next(iter(dataloader))
lowres_batch = lowres_batch.cuda()

generated = superres_maskgit.generate(
    dna_coords=coords_batch,     # ✅ pass to your Enformer encoder
    cond_images=lowres_batch,
    cond_scale=3.0
)



print("Generated shape:", generated.shape)  # [B, 1, 512, 512]