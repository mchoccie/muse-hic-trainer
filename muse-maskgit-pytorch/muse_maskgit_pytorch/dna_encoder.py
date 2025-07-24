# dna_encoder.py
import torch, json, numpy as np
from typing import List, Tuple
from pathlib import Path
from enformer_pytorch import Enformer
from kipoiseq.transforms.functional import one_hot_dna
from pyfaidx import Fasta


SEQ_LEN = 196_608             # Enformer expects exactly this length
ONEHOT_SEQ_LEN = 131072  # One-hot encoder expects this length
EMB_LAYER = -2                # last-2 transformer block is common
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

class EnformerEncoder:
    """
    → .encode(seqs)  returns tensor [B, L_emb, D] suitable for cross-attention
    """
    def __init__(self, enformer_dir: str, genome_fasta: str):
        # load model
        self.model = Enformer.from_hparams().to(DEVICE).eval()
        state = torch.load(Path(enformer_dir, "pytorch_model.bin"), map_location=DEVICE)
        self.model.load_state_dict(state, strict=False)

        # fasta
        self.genome = Fasta(genome_fasta)

    # ---------------------------------------------------------
    def _prep_seq(self, chrom: str, start: int, end: int) -> torch.Tensor:
        """fetch hg19 sequence, centre-crop / pad to 196 608 bp, one-hot encode"""
        seq = self.genome[chrom][start:end].seq.upper().replace('N', 'A')
        if len(seq) < SEQ_LEN:
            pad = (SEQ_LEN - len(seq)) // 2
            seq = 'A' * pad + seq + 'A' * pad
        elif len(seq) > SEQ_LEN:
            off = (len(seq) - SEQ_LEN) // 2
            seq = seq[off:off+SEQ_LEN]
        return torch.tensor(one_hot_dna(seq).astype(np.float32))     # [SEQ_LEN, 4]

    # ---------------------------------------------------------
    @torch.no_grad()
    def encode(self, coords: List[tuple]) -> torch.Tensor:
        """
        coords :  [(chrom, start, end), ...]  (0-based, half-open)
        returns : [B, L_emb, D]  (here L_emb == 1536,  D == 1536 for official Enformer)
        """
        # collect one-hot
        batch = torch.stack([self._prep_seq(*c) for c in coords]).to(DEVICE)  # [B, 196k, 4]

        # forward – get hidden states at all layers
        _, hidden = self.model(batch, return_embeddings=True)

        # hidden is list; we want the chosen layer
        print("HIDEEN SHAPPEEE: " + str(hidden.shape))
        embeds = hidden                                # [B, 1536, 1536]
        print(embeds.shape)
        return embeds


class OneHotDNAEncoder:
    """
    → .encode(seqs)  returns tensor [B, L, 4] suitable for projection or pooling
    """
    def __init__(self, genome_fasta: str):
        self.genome = Fasta(genome_fasta)

    def _prep_seq(self, chrom: str, start: int, end: int) -> torch.Tensor:
        """Fetch and one-hot encode sequence, padding or cropping as needed"""
        seq = self.genome[chrom][start:end].seq.upper().replace('N', 'A')

        # Pad or center-crop to fixed length
        if len(seq) < ONEHOT_SEQ_LEN:
            pad_total = ONEHOT_SEQ_LEN - len(seq)
            left = pad_total // 2
            right = pad_total - left
            seq = 'A' * left + seq + 'A' * right
        elif len(seq) > ONEHOT_SEQ_LEN:
            offset = (len(seq) - ONEHOT_SEQ_LEN) // 2
            seq = seq[offset:offset + ONEHOT_SEQ_LEN]

        onehot = one_hot_dna(seq).astype(np.float32)   # [L, 4]
        return torch.tensor(onehot)  # shape: [L, 4]

    @torch.no_grad()
    def encode(self, coords: List[Tuple[str, int, int]]) -> torch.Tensor:
        batch = [self._prep_seq(*c) for c in coords]  # list of [L, 4]
        stacked = torch.stack(batch).to(DEVICE)          # [B, L, 4]
        pooled = torch.nn.functional.avg_pool1d(
        input=stacked.permute(0, 2, 1),  # [B, 4, L]
        kernel_size=128,
        stride=128
        ).permute(0, 2, 1)  # [B, L//128, 4]

        return pooled