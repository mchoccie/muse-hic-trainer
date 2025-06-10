from pyfaidx import Fasta
from enformer_pytorch import Enformer
from enformer_pytorch import from_pretrained
from kipoiseq.transforms.functional import one_hot_dna
import torch
import numpy as np
import pandas as pd
import json
# model = from_pretrained('EleutherAI/enformer-official-rough')
# print(model)
# model.save_pretrained('./enformer_local', safe_serialization=False)

# Load config
with open('enformer_local/config.json') as f:
    config = json.load(f)

model = Enformer.from_hparams()

# Load weights
model.load_state_dict(torch.load('enformer_local/pytorch_model.bin', map_location='cpu'))

# Set to evaluation mode
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
        output, embeddings = model(x, return_embeddings = True)
        print(len(embeddings[0]))
        # print(f"Before permutation: {y['human'].shape}")  # [1, 896, 5313]
        # y['human'] = y['human'].permute(0, 2, 1)
        # print(f"After permutation: {y['human'].shape}")   # [1, 5313, 896]
        # print(f"The track we're concerned with: {y['human'][0][5110].shape}")   # [1, 5313, 896]
        print(f"{chrom}:{start}-{end} -> output embeddings: {embeddings}")
        break