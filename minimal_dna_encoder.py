#!/usr/bin/env python3
"""
Minimal DNA encoder for testing CUDA indexing issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from pyfaidx import Fasta

class MinimalDNAEncoder(nn.Module):
    """
    Minimal DNA encoder that doesn't require kipoiseq
    """
    def __init__(self, genome_fasta: str, embedding_dim: int = 256, max_seq_len: int = 10000):
        super().__init__()
        self.genome = Fasta(genome_fasta)
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Simple one-hot to embedding projection
        self.onehot_proj = nn.Linear(4, embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def _one_hot_encode(self, seq: str) -> np.ndarray:
        """Simple one-hot encoding without kipoiseq"""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        onehot = np.zeros((len(seq), 4), dtype=np.float32)
        
        for i, base in enumerate(seq):
            if base in mapping:
                onehot[i, mapping[base]] = 1.0
        
        return onehot
    
    def _prep_sequence(self, chrom: str, start: int, end: int) -> torch.Tensor:
        """Prepare sequence with simple one-hot encoding"""
        try:
            seq = self.genome[chrom][start:end].seq.upper().replace('N', 'A')
        except Exception as e:
            print(f"Error reading sequence for {chrom}:{start}-{end}: {e}")
            # Return a dummy sequence if there's an error
            seq = 'A' * (end - start)
        
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
        onehot = self._one_hot_encode(seq)
        return torch.tensor(onehot, dtype=torch.float32)
    
    @torch.no_grad()
    def encode(self, coords: List[Tuple[str, int, int]]) -> torch.Tensor:
        """Encode DNA coordinates to embeddings"""
        print(f"Encoding {len(coords)} coordinates...")
        
        sequences = []
        for i, (chrom, start, end) in enumerate(coords):
            print(f"Processing coordinate {i+1}/{len(coords)}: {chrom}:{start}-{end}")
            seq_tensor = self._prep_sequence(chrom, start, end)
            sequences.append(seq_tensor)
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences)
        print(f"Max sequence length: {max_len}")
        
        padded_seqs = []
        for seq in sequences:
            if len(seq) < max_len:
                # Pad with zeros
                pad_size = max_len - len(seq)
                padded = F.pad(seq, (0, 0, 0, pad_size), value=0)
            else:
                padded = seq[:max_len]
            padded_seqs.append(padded)
        
        # Stack and project
        print(f"Stacking {len(padded_seqs)} sequences...")
        batch = torch.stack(padded_seqs)  # [B, L, 4]
        print(f"Batch shape: {batch.shape}")
        
        # Move to device if needed
        if batch.device != self.onehot_proj.weight.device:
            batch = batch.to(self.onehot_proj.weight.device)
        
        print(f"Projecting to embeddings...")
        x = self.onehot_proj(batch)  # [B, L, D]
        print(f"Projected shape: {x.shape}")
        
        # Global average pooling
        pooled = torch.mean(x, dim=1)  # [B, D]
        print(f"Pooled shape: {pooled.shape}")
        
        # Output projection
        output = self.output_proj(pooled)  # [B, D]
        print(f"Output shape: {output.shape}")
        
        return output.unsqueeze(1)  # [B, 1, D]

def test_minimal_encoder():
    """Test the minimal DNA encoder"""
    
    print("="*60)
    print("TESTING MINIMAL DNA ENCODER")
    print("="*60)
    
    # Test configuration
    genome_fasta = '/scratch/rnd-rojas/Manan/hg19.fa'
    embedding_dim = 256
    
    # Check if genome file exists
    import os
    if not os.path.exists(genome_fasta):
        print(f"✗ Genome file not found: {genome_fasta}")
        return False
    
    print(f"✓ Genome file found: {genome_fasta}")
    
    # Test coordinates (very small regions)
    test_coords = [
        ('chr1', 1000000, 1001000),  # 1kb region
        ('chr2', 2000000, 2001000),  # 1kb region
    ]
    
    try:
        print(f"Creating MinimalDNAEncoder...")
        encoder = MinimalDNAEncoder(
            genome_fasta=genome_fasta,
            embedding_dim=embedding_dim,
            max_seq_len=1000  # Very small for testing
        )
        print(f"✓ Encoder created successfully")
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        encoder = encoder.to(device)
        print(f"✓ Encoder moved to device")
        
        # Test encoding
        print(f"Testing encoding with coordinates: {test_coords}")
        embeddings = encoder.encode(test_coords)
        
        print(f"✓ Encoding successful!")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Expected shape: ({len(test_coords)}, 1, {embedding_dim})")
        print(f"  Output range: [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]")
        print(f"  Output mean: {embeddings.mean().item():.4f}")
        print(f"  Output std: {embeddings.std().item():.4f}")
        
        # Check for NaN or inf values
        if torch.isnan(embeddings).any():
            print("✗ WARNING: NaN values detected in output!")
            return False
        if torch.isinf(embeddings).any():
            print("✗ WARNING: Inf values detected in output!")
            return False
        
        print("✓ No NaN or inf values detected")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_encoder()
    
    print("\n" + "="*60)
    if success:
        print("🎉 MINIMAL ENCODER TEST PASSED!")
    else:
        print("❌ MINIMAL ENCODER TEST FAILED!")
    print("="*60) 