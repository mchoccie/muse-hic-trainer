# Save this as: /scratch/rnd-rojas/Manan/muse-maskgit-pytorch/muse_maskgit_pytorch/improved_dna_encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from pyfaidx import Fasta
from kipoiseq.transforms.functional import one_hot_dna
import re
from collections import Counter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class KmerDNAEncoder(nn.Module):
    """
    K-mer based DNA encoder that captures local sequence patterns
    More biologically relevant than simple one-hot encoding
    """
    def __init__(self, genome_fasta: str, kmer_size: int = 6, embedding_dim: int = 256, max_seq_len: int = 131072):
        super().__init__()
        self.genome = Fasta(genome_fasta)
        self.kmer_size = kmer_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # K-mer vocabulary (4^k possible k-mers)
        self.vocab_size = 4 ** kmer_size
        self.kmer_embeddings = nn.Embedding(self.vocab_size, embedding_dim)
        
        # Positional encoding for sequence position
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len // 64, embedding_dim))
        
        # Transformer for sequence modeling
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def _kmer_to_index(self, kmer: str) -> int:
        """Convert k-mer string to index"""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        index = 0
        for i, base in enumerate(kmer):
            index += mapping[base] * (4 ** i)
        return index
    
    def _extract_kmers(self, seq: str) -> List[int]:
        """Extract k-mer indices from sequence"""
        kmers = []
        for i in range(len(seq) - self.kmer_size + 1):
            kmer = seq[i:i + self.kmer_size]
            kmers.append(self._kmer_to_index(kmer))
        return kmers
    
    def _prep_sequence(self, chrom: str, start: int, end: int) -> torch.Tensor:
        """Prepare sequence and extract k-mers"""
        seq = self.genome[chrom][start:end].seq.upper().replace('N', 'A')
        
        # Pad or crop to max_seq_len
        if len(seq) < self.max_seq_len:
            pad_total = self.max_seq_len - len(seq)
            left = pad_total // 2
            right = pad_total - left
            seq = 'A' * left + seq + 'A' * right
        else:
            offset = (len(seq) - self.max_seq_len) // 2
            seq = seq[offset:offset + self.max_seq_len]
        
        # Extract k-mers
        kmer_indices = self._extract_kmers(seq)
        return torch.tensor(kmer_indices, dtype=torch.long)
    
    @torch.no_grad()
    def encode(self, coords: List[Tuple[str, int, int]]) -> torch.Tensor:
        """Encode DNA coordinates to embeddings"""
        sequences = [self._prep_sequence(chrom, start, end) for chrom, start, end in coords]
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences)
        padded_seqs = []
        
        for seq in sequences:
            if len(seq) < max_len:
                padded = F.pad(seq, (0, max_len - len(seq)), value=0)
            else:
                padded = seq[:max_len]
            padded_seqs.append(padded)
        
        # Stack and embed
        batch = torch.stack(padded_seqs).to(self.kmer_embeddings.weight.device)  # [B, L]
        embeddings = self.kmer_embeddings(batch)  # [B, L, D]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :embeddings.size(1), :]
        embeddings = embeddings + pos_enc
        
        # Apply transformer
        encoded = self.transformer(embeddings)  # [B, L, D]
        
        # Global pooling and projection
        pooled = torch.mean(encoded, dim=1)  # [B, D]
        output = self.output_proj(pooled)  # [B, D]
        
        return output.unsqueeze(1)  # [B, 1, D] for compatibility

class MotifDNAEncoder(nn.Module):
    """
    DNA encoder based on known transcription factor binding motifs
    More biologically interpretable than k-mers
    """
    def __init__(self, genome_fasta: str, embedding_dim: int = 256, max_seq_len: int = 131072):
        super().__init__()
        self.genome = Fasta(genome_fasta)
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Common TF binding motifs (simplified)
        self.motifs = {
            'TATA': 'TATAAA',
            'CAAT': 'CAAT',
            'GC': 'GCGC',
            'E2F': 'TTTCGCGC',
            'SP1': 'GGGCGG',
            'AP1': 'TGACTCA',
            'CREB': 'TGACGTCA',
            'NFKB': 'GGGRNNYYCC',
            'OCT1': 'ATGCAAAT',
            'ETS': 'GGAAT'
        }
        
        # Motif embedding layer
        self.motif_embeddings = nn.Embedding(len(self.motifs), embedding_dim)
        
        # Convolutional layers for motif detection
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(4, 64, kernel_size=4, padding=1),
            nn.Conv1d(64, 128, kernel_size=4, padding=1),
            nn.Conv1d(128, 256, kernel_size=4, padding=1),
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def _find_motifs(self, seq: str) -> List[int]:
        """Find motif occurrences in sequence"""
        motif_counts = []
        
        for motif_name, motif_seq in self.motifs.items():
            # Simple pattern matching (could be improved with fuzzy matching)
            count = len(re.findall(motif_seq, seq))
            motif_counts.append(count)
        
        return motif_counts
    
    def _prep_sequence(self, chrom: str, start: int, end: int) -> Tuple[torch.Tensor, List[int]]:
        """Prepare sequence and find motifs"""
        seq = self.genome[chrom][start:end].seq.upper().replace('N', 'A')
        
        # Pad or crop to max_seq_len
        if len(seq) < self.max_seq_len:
            pad_total = self.max_seq_len - len(seq)
            left = pad_total // 2
            right = pad_total - left
            seq = 'A' * left + seq + 'A' * right
        else:
            offset = (len(seq) - self.max_seq_len) // 2
            seq = seq[offset:offset + self.max_seq_len]
        
        # One-hot encode for CNN
        onehot = one_hot_dna(seq).astype(np.float32)
        
        # Find motifs
        motif_counts = self._find_motifs(seq)
        
        return torch.tensor(onehot), motif_counts
    
    @torch.no_grad()
    def encode(self, coords: List[Tuple[str, int, int]]) -> torch.Tensor:
        """Encode DNA coordinates to embeddings"""
        sequences = [self._prep_sequence(chrom, start, end) for chrom, start, end in coords]
        
        batch_onehot = []
        batch_motifs = []
        
        for onehot, motifs in sequences:
            batch_onehot.append(onehot)
            batch_motifs.append(motifs)
        
        # Process one-hot sequences with CNN
        onehot_batch = torch.stack(batch_onehot).to(self.conv_layers[0].weight.device)  # [B, L, 4]
        onehot_batch = onehot_batch.permute(0, 2, 1)  # [B, 4, L]
        
        # Apply convolutional layers
        x = onehot_batch
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool1d(x, kernel_size=2)
        
        # Global average pooling
        cnn_features = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, 256]
        
        # Process motif information
        motif_tensor = torch.tensor(batch_motifs, dtype=torch.long).to(self.motif_embeddings.weight.device)  # [B, num_motifs]
        motif_embeddings = self.motif_embeddings(motif_tensor)  # [B, num_motifs, D]
        
        # Apply attention to motif embeddings
        attended_motifs, _ = self.attention(motif_embeddings, motif_embeddings, motif_embeddings)
        motif_features = torch.mean(attended_motifs, dim=1)  # [B, D]
        
        # Combine CNN and motif features
        combined = cnn_features + motif_features
        output = self.output_proj(combined)  # [B, D]
        
        return output.unsqueeze(1)  # [B, 1, D]

class EfficientDNAEncoder(nn.Module):
    """
    Lightweight DNA encoder that balances efficiency and biological relevance
    Uses learned embeddings for DNA patterns - RECOMMENDED for your use case
    """
    def __init__(self, genome_fasta: str, embedding_dim: int = 256, max_seq_len: int = 131072, window_size: int = 1024):
        super().__init__()
        self.genome = Fasta(genome_fasta)
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        
        # DNA base embeddings
        self.base_embeddings = nn.Embedding(4, embedding_dim // 4)  # A, C, G, T
        
        # Local pattern detector
        self.pattern_conv = nn.Sequential(
            nn.Conv1d(embedding_dim // 4, embedding_dim // 2, kernel_size=8, padding=3),
            nn.ReLU(),
            nn.Conv1d(embedding_dim // 2, embedding_dim, kernel_size=8, padding=3),
            nn.ReLU(),
        )
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def _prep_sequence(self, chrom: str, start: int, end: int) -> torch.Tensor:
        """Prepare sequence with efficient encoding"""
        seq = self.genome[chrom][start:end].seq.upper().replace('N', 'A')
        
        # Pad or crop to max_seq_len
        if len(seq) < self.max_seq_len:
            pad_total = self.max_seq_len - len(seq)
            left = pad_total // 2
            right = pad_total - left
            seq = 'A' * left + seq + 'A' * right
        else:
            offset = (len(seq) - self.max_seq_len) // 2
            seq = seq[offset:offset + self.max_seq_len]
        
        # Convert to indices (A=0, C=1, G=2, T=3)
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        indices = [mapping[base] for base in seq]
        
        return torch.tensor(indices, dtype=torch.long)
    
    @torch.no_grad()
    def encode(self, coords: List[Tuple[str, int, int]]) -> torch.Tensor:
        """Encode DNA coordinates to embeddings"""
        sequences = [self._prep_sequence(chrom, start, end) for chrom, start, end in coords]
        
        # Pad sequences
        max_len = max(len(seq) for seq in sequences)
        padded_seqs = []
        
        for seq in sequences:
            if len(seq) < max_len:
                padded = F.pad(seq, (0, max_len - len(seq)), value=0)
            else:
                padded = seq[:max_len]
            padded_seqs.append(padded)
        
        # Stack and embed
        batch = torch.stack(padded_seqs).to(self.base_embeddings.weight.device)  # [B, L]
        embeddings = self.base_embeddings(batch)  # [B, L, D//4]
        
        # Apply pattern detection
        x = embeddings.permute(0, 2, 1)  # [B, D//4, L]
        x = self.pattern_conv(x)  # [B, D, L]
        
        # Global pooling
        pooled = self.global_pool(x).squeeze(-1)  # [B, D]
        output = self.output_proj(pooled)  # [B, D]
        
        return output.unsqueeze(1)  # [B, 1, D]

class SimpleDNAEncoder(nn.Module):
    """
    Simple DNA encoder that's compatible with the existing pipeline
    Uses one-hot encoding with better pooling strategy
    """
    def __init__(self, genome_fasta: str, embedding_dim: int = 256, max_seq_len: int = 131072):
        super().__init__()
        self.genome = Fasta(genome_fasta)
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # One-hot to embedding projection
        self.onehot_proj = nn.Linear(4, embedding_dim // 4)
        
        # Multi-scale pooling
        self.pool_sizes = [64, 128, 256, 512]
        self.pool_layers = nn.ModuleList([
            nn.AdaptiveAvgPool1d(size) for size in self.pool_sizes
        ])
        
        # Feature fusion
        total_features = sum((embedding_dim // 4) * p for p in self.pool_sizes)
        print(f"Total features: {total_features}")
        self.fusion_layer = nn.Linear(total_features, embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def _prep_sequence(self, chrom: str, start: int, end: int) -> torch.Tensor:
        """Prepare sequence with one-hot encoding"""
        seq = self.genome[chrom][start:end].seq.upper().replace('N', 'A')
        
        # Pad or crop to max_seq_len
        if len(seq) < self.max_seq_len:
            pad_total = self.max_seq_len - len(seq)
            left = pad_total // 2
            right = pad_total - left
            seq = 'A' * left + seq + 'A' * right
        else:
            offset = (len(seq) - self.max_seq_len) // 2
            seq = seq[offset:offset + self.max_seq_len]
        
        # One-hot encode
        onehot = one_hot_dna(seq).astype(np.float32)
        return torch.tensor(onehot, dtype=torch.float32)
    
    @torch.no_grad()
    def encode(self, coords: List[Tuple[str, int, int]]) -> torch.Tensor:
        """Encode DNA coordinates to embeddings"""
        sequences = [self._prep_sequence(chrom, start, end) for chrom, start, end in coords]
        
        # Pad sequences
        max_len = max(len(seq) for seq in sequences)
        padded_seqs = []
        
        for seq in sequences:
            if len(seq) < max_len:
                padded = F.pad(seq, (0, 0, 0, max_len - len(seq)), value=0)
            else:
                padded = seq[:max_len]
            padded_seqs.append(padded)
        
        # Stack and project
        batch = torch.stack(padded_seqs).to(self.onehot_proj.weight.device)  # [B, L, 4]
        x = self.onehot_proj(batch)  # [B, L, D//4]
        
        # Multi-scale pooling
        pooled_features = []
        x_permuted = x.permute(0, 2, 1)  # [B, D//4, L]
        
        for pool_layer in self.pool_layers:
            pooled = pool_layer(x_permuted)  # [B, D//4, pool_size]
            pooled_flat = pooled.flatten(1)  # [B, D//4 * pool_size]
            pooled_features.append(pooled_flat)
        
        # Concatenate all pooled features
        combined = torch.cat(pooled_features, dim=1)  # [B, total_features]
        
        # Fusion and output
        fused = self.fusion_layer(combined)  # [B, D]
        output = self.output_proj(fused)  # [B, D]
        
        return output.unsqueeze(1)  # [B, 1, D]

# Factory function to create encoders
def create_dna_encoder(encoder_type: str = 'simple', genome_fasta: str = None, **kwargs):
    """Create DNA encoder based on type"""
    if genome_fasta is None:
        raise ValueError("genome_fasta path is required")
        
    if encoder_type == 'kmer':
        return KmerDNAEncoder(genome_fasta, **kwargs)
    elif encoder_type == 'motif':
        return MotifDNAEncoder(genome_fasta, **kwargs)
    elif encoder_type == 'efficient':
        return EfficientDNAEncoder(genome_fasta, **kwargs)
    elif encoder_type == 'simple':
        return SimpleDNAEncoder(genome_fasta, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")