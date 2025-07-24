#!/usr/bin/env python
import torch, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
from muse_maskgit_pytorch.dna_encoder import EnformerEncoder
from muse_maskgit_pytorch.muse_maskgit_pytorch import generate_from_dna

# ------------------------------------------------------------------  paths
root = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch"
lowres_np_path  = f"{root}/lowres_dataset.npy"       # [N, 1, 256, 256]
highres_np_path = f"{root}/highres_dataset.npy"      # [N, 1, 512, 512]
coords_path     = f"{root}/hic_window_coords.npy"

# ------------------------------------------------------------------  load arrays
lowres_np   = np.load(lowres_np_path,   mmap_mode="r")
highres_np  = np.load(highres_np_path,  mmap_mode="r")
coords      = [tuple(c) for c in np.load(coords_path, allow_pickle=True).tolist()]

assert len(coords) == len(lowres_np) == len(highres_np)

# ------------------------------------------------------------------  dataset / loader
class HiCDataset(Dataset):
    def __init__(self, low, high, coords):
        self.low, self.high, self.coords = low, high, coords
    def __len__(self): return len(self.coords)
    def __getitem__(self, idx):
        return ( torch.from_numpy(self.low [idx]).float(),   # [1,256,256]
                 torch.from_numpy(self.high[idx]).float(),   # [1,512,512]
                 self.coords[idx] )

def collate_fn(batch):
    lows, highs, coords = zip(*batch)
    return torch.stack(lows), torch.stack(highs), list(coords)

loader = DataLoader(HiCDataset(lowres_np, highres_np, coords),
                    batch_size=8, shuffle=False, collate_fn=collate_fn)

lowres_batch, highres_batch, coords_batch = next(iter(loader))
lowres_batch  = lowres_batch.cuda()            # [B,1,256,256]
highres_batch = highres_batch.cuda()

# ------------------------------------------------------------------  DNA encoder + model
dna_encoder = EnformerEncoder('/scratch/rnd-rojas/Manan/enformer_local',
                              '/scratch/rnd-rojas/Manan/hg19.fa')

vae = VQGanVAE(dim=256, codebook_size=1024, use_vgg_and_gan=False).cuda()
vae.load('/scratch/rnd-rojas/Manan/baseResultsHighresolution/vae.49000.pt')

transformer = MaskGitTransformer(num_tokens=1024, dim=512, seq_len=1024,
                                 depth=8, dna_encoder=dna_encoder).cuda()

superres_maskgit = MaskGit(
        vae             = vae,
        transformer     = transformer,
        image_size      = 512,
        cond_image_size = 256,     # <-- very important for SR
        cond_drop_prob  = 0.
).cuda()

superres_maskgit.load_state_dict(
    torch.load("/scratch/rnd-rojas/Manan/maskgit_highres.pt")
)

# ------------------------------------------------------------------  generate
generated = generate_from_dna(superres_maskgit,
                              dna_coords=coords_batch,
                              cond_images=lowres_batch,
                              cond_scale=3.0)          # [B,1,512,512]

print("generated:", generated.shape)

# ---------------------------------------------------------------  print arrays
np.set_printoptions(edgeitems=4, linewidth=140, suppress=True)   # tidy console

for i in range(min(8, generated.size(0))):
    arr = generated[i, 0].cpu().numpy()        # (512, 512)  float32
    print(f"\n=== Predicted map #{i}  {coords_batch[i]}  ===")
    print(arr)                                 # huge!  remove if undesired

    # optional quick summary per map
    print("  →  min {:.3f} | max {:.3f} | mean {:.3f} | std {:.3f}"
          .format(arr.min(), arr.max(), arr.mean(), arr.std()))

# ------------------------------------------------------------------  quick stats
for tag, arr in [("low256", lowres_batch[:,0]),
                 ("true512", highres_batch[:,0]),
                 ("gen512", generated[:,0])]:
    flat = arr.cpu().view(-1)
    print(f"{tag:8s}  min={flat.min():6.2f}  max={flat.max():6.2f}  mean={flat.mean():6.2f}")

# ------------------------------------------------------------------  plot
B = generated.size(0)
gen_np   = generated[:,0].cpu().numpy()
truth_np = highres_batch[:,0].cpu().numpy()

fig, axes = plt.subplots(2, B, figsize=(2.5*B, 5))
for i in range(B):
    # generated
    ax = axes[0, i]
    im = ax.imshow(np.log(gen_np[i]), cmap='magma')
    ax.set_title(f"GEN\n{coords_batch[i][0]}:{coords_batch[i][1]}")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    # ground-truth
    ax = axes[1, i]
    im = ax.imshow(truth_np[i], cmap='magma', interpolation='nearest')
    ax.set_title("GT")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

plt.tight_layout()
plt.savefig("gen_vs_gt_batch0.png", dpi=150)
plt.show()
