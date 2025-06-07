from pyfaidx import Fasta
from enformer_pytorch import Enformer
from kipoiseq.transforms.functional import one_hot_dna
import torch
import numpy as np

model = Enformer.from_hparams()
print(model)
model.eval()


hg19 = Fasta('hg19.fa')

def get_onehot_sequence(seq, target_len=196608):
    seq = seq.upper().replace('N', 'A')
    if len(seq) < target_len:
        pad_len = (target_len - len(seq)) // 2
        seq = 'A' * pad_len + seq + 'A' * pad_len
    elif len(seq) > target_len:
        start = (len(seq) - target_len) // 2
        seq = seq[start:start+target_len]
    return one_hot_dna(seq).astype(np.float32)

coordinates = np.load("/scratch/rnd-rojas/Manan/hic_window_coords.npy", allow_pickle=True)

for chrom, start, end in coordinates[:5]:
    seq = hg19[chrom][start:end].seq
    onehot = get_onehot_sequence(seq)
    print(onehot.shape)
    x = torch.tensor(onehot).unsqueeze(0)
    
    with torch.no_grad():
        print("This is the FINAL tensor passed in:")
        print("x.shape:", x.shape)
        print("x.dtype:", x.dtype)
        print("x.device:", x.device)
        print("type(x):", type(x))
        y = model(x)
        print(f"{chrom}:{start}-{end} -> output shape: {y['human'].shape}")