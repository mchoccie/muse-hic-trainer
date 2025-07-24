import torch
from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
from muse_maskgit_pytorch.dna_encoder import EnformerEncoder, OneHotDNAEncoder
import numpy as np

# ------------------------------------------------------------------
# 1)  load the files
# ------------------------------------------------------------------

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def custom_collate(batch):
    lowres, highres, coords = zip(*batch)
    return (
        torch.stack(lowres),
        torch.stack(highres),
        list(coords),  # this preserves coords as a list of tuples
            # this preserves coords as a list of tuples
    )

class HiCDataset(Dataset):
    def __init__(self, lowres_np, highres_np, coords):
        self.lowres_np = lowres_np
        self.highres_np = highres_np
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        lowres = torch.from_numpy(self.lowres_np[idx]).float()
        highres = torch.from_numpy(self.highres_np[idx]).float()
        coords = tuple(self.coords[idx])  # ('chr1', start, end)
        return lowres, highres, coords
hic_path_lowres   = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/lowres_dataset.npy"
hic_path_highres  = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/highres_dataset.npy"
coords_path = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_window_coords.npy"

# Hi-C maps       – shape should be [N, 1, 256, 256]
hic_np_lowres = np.load(hic_path_lowres, mmap_mode="r")         # mmap saves RAM, remove if you plan to edit
hic_np_highres = np.load(hic_path_highres, mmap_mode="r")
print("Hi-C numpy  shape :", hic_np_lowres.shape, hic_np_lowres.dtype)
num_steps = 1000


# window coordinates – saved as a pickled object array of tuples
coords = np.load(coords_path, allow_pickle=True)
coords = [tuple(c) for c in coords.tolist()]                 # convert to plain list
print("coord list len:", len(coords))
print("first coord   :", coords[0])               # e.g. ('chr1', 0, 12800000)
print(hic_np_lowres.shape[0])
assert len(coords) == hic_np_lowres.shape[0], "mismatch in #windows"
dataset = HiCDataset(hic_np_lowres, hic_np_highres, coords)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate)

# ------------------------------------------------------------------
# 2)  choose a batch (optional) and move to torch
# ------------------------------------------------------------------
# batch_idx     = np.arange(4)          # first 4 windows for example
# batch_images_lowres  = torch.from_numpy(hic_np_lowres[batch_idx]).float().cuda()  # [B, 1, 256, 256]
# batch_images_highres = torch.from_numpy(hic_np_highres[batch_idx]).float().cuda()  # [B, 1, 256, 256]
# batch_coords  = [coords[i] for i in batch_idx]               # list of tuples

# print("batch_images shape:", batch_images_lowres.shape)
# print("batch_coords      :", batch_coords)
enformer_dir = '/scratch/rnd-rojas/Manan/enformer_local'
genome_fasta = '/scratch/rnd-rojas/Manan/hg19.fa'

dna_enc = OneHotDNAEncoder(genome_fasta)
vaeBase = VQGanVAE(
    dim = 256,
    codebook_size = 1024,
    use_vgg_and_gan = False
).cuda()

vaeHighres = VQGanVAE(
    dim = 256,
    codebook_size = 1024,
    use_vgg_and_gan = False
).cuda()

vaeBase.load('/scratch/rnd-rojas/Manan/vqgan_25kb_ckpts/vae.49000.pt') # you will want to load the exponentially moving averaged VAE
vaeHighres.load('/scratch/rnd-rojas/Manan/baseResultsHighresolution/vae.49000.pt') # you will want to load the exponentially moving averaged VAE
# ------------------------------------------------------------------
# 3)  create the MaskGit model
# ------------------------------------------------------------------
transformerLowRes = MaskGitTransformer(
    num_tokens = 1024,   # codebook size
    dim        = 512,
    seq_len    = 1024,
    depth      = 8,
    dna_encoder = dna_enc,   # ← plug in here
)

transformerHighRes = MaskGitTransformer(
    num_tokens = 1024,   # codebook size
    dim        = 512,
    seq_len    = 1024,
    depth      = 8,
    dna_encoder = dna_enc,   # ← plug in here
)


transformerLowRes = transformerLowRes.cuda()  # 👈 Move it AFTER initialization
transformerHighRes = transformerHighRes.cuda()  # 👈 Move it AFTER initialization

maskgit = MaskGit(
    vae           = vaeBase,
    transformer   = transformerLowRes,
    image_size    = 256,
    cond_image_size = None,  # not doing SR here
)

superres_maskgit = MaskGit(
    vae = vaeHighres,
    transformer = transformerHighRes,
    cond_drop_prob = 0.25,
    image_size = 512,                     # larger image size
    cond_image_size = 256,                # conditioning image size <- this must be set
).cuda()

optimizer = torch.optim.AdamW(maskgit.parameters(), lr=1e-4)
optimizer_superres = torch.optim.AdamW(superres_maskgit.parameters(), lr=1e-4)

# /scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_dataset_50kb.npy -- these are the Hi-C maps
# /scratch/rnd-rojas/Manan/hic_window_coords.npy -- these are the coordinates of the Hi-C windows


# Need to train both maskgit models separately
# loss = maskgit(
#     batch_images,                         # [B, 1, 256, 256] Hi-C maps
#     dna_coords = batch_coords,      # list[ (chrom,start,end) ] len==B
# )

superres_maskgit.train()
maskgit.train()
for step, (lowres_batch, highres_batch, coords) in enumerate(dataloader):
    lowres_batch = lowres_batch.cuda()      # shape: [B, 1, 256, 256]
    highres_batch = highres_batch.cuda()
    #loss = superres_maskgit(highres_batch, dna_coords=coords, cond_images=lowres_batch)
    loss = maskgit(lowres_batch, dna_coords=coords)
    #optimizer_superres.zero_grad()
    optimizer.zero_grad()
    loss.backward()
    #optimizer_superres.step()
    optimizer.step()

    print(f"[Step {step}] loss = {loss.item():.4f}")

torch.save(maskgit.state_dict(), "maskgit_lowres.pt")

