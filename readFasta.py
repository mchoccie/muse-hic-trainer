from pyfaidx import Fasta
import numpy as np

# Load hg19 genome (do this ONCE)
genome = Fasta("hg19.fa")

# Load coordinates
coords = np.load("hic_window_coords.npy", allow_pickle=True)

for i, (chrom, start, end) in enumerate(coords):
    print(f"Window {i}: {chrom}:{start}-{end}")