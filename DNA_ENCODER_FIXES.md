# DNA Encoder Fixes and Improvements

## 🚨 **Issues Fixed**

### 1. **Index Out of Bounds Error**
**Problem**: The original `EfficientDNAEncoder` was trying to embed sequences that were too long for the embedding layer, causing CUDA indexing errors.

**Solution**: 
- Fixed device handling to use the encoder's own device instead of global `DEVICE`
- Added proper sequence length management
- Created a new `SimpleDNAEncoder` as the default option

### 2. **Missing Method Error**
**Problem**: The code was calling `self.transformer.encode_dna(dna_coords)` which doesn't exist.

**Solution**: 
- Fixed to use `self.transformer.dna_encoder.encode(dna_coords)` directly
- Added proper device synchronization

### 3. **Device Mismatch Issues**
**Problem**: DNA encoder and context tensors were on different devices.

**Solution**:
- Added automatic device synchronization in the transformer forward method
- Ensured all tensors are on the same device before processing

## 🔧 **Improved DNA Encoders**

### 1. **SimpleDNAEncoder** (RECOMMENDED)
```python
# Default encoder - most stable and efficient
dna_encoder = create_dna_encoder(
    encoder_type='simple',
    genome_fasta='/path/to/genome.fa',
    embedding_dim=256
)
```

**Features**:
- One-hot encoding with multi-scale pooling
- No indexing issues
- Fast and memory efficient
- Compatible with existing pipeline

### 2. **EfficientDNAEncoder**
```python
# Lightweight with learned patterns
dna_encoder = create_dna_encoder(
    encoder_type='efficient',
    genome_fasta='/path/to/genome.fa',
    embedding_dim=256
)
```

**Features**:
- Base-level embeddings with convolutional pattern detection
- Better biological representation than simple one-hot
- Fixed indexing issues

### 3. **KmerDNAEncoder**
```python
# K-mer based encoding
dna_encoder = create_dna_encoder(
    encoder_type='kmer',
    genome_fasta='/path/to/genome.fa',
    embedding_dim=256,
    kmer_size=6
)
```

**Features**:
- Captures local sequence patterns
- Transformer-based sequence modeling
- More biologically relevant

### 4. **MotifDNAEncoder**
```python
# Transcription factor motif based
dna_encoder = create_dna_encoder(
    encoder_type='motif',
    genome_fasta='/path/to/genome.fa',
    embedding_dim=256
)
```

**Features**:
- Based on known TF binding motifs
- CNN + attention architecture
- Most biologically interpretable

## 🚀 **How to Use**

### 1. **Update Your Training Pipeline**
```python
# In muse_pipeline_improved.py
from muse_maskgit_pytorch.improved_dna_encoders import create_dna_encoder

# Create DNA encoder
dna_encoder = create_dna_encoder(
    encoder_type='simple',  # or 'efficient', 'kmer', 'motif'
    genome_fasta='/scratch/rnd-rojas/Manan/hg19.fa',
    embedding_dim=256
)

# Use in transformer
transformer = MaskGitTransformer(
    num_tokens=1024,
    dim=768,
    seq_len=1024,
    depth=12,
    heads=12,
    dim_head=64,
    dna_encoder=dna_encoder,  # Pass the encoder here
    self_cond=True,
)
```

### 2. **Test the Encoders**
```bash
# Run the test script
python test_dna_encoders.py
```

This will test all encoders and verify they work correctly.

### 3. **Start Training**
```bash
# Use the improved pipeline
python muse_pipeline_improved.py
```

## 📊 **Performance Comparison**

| Encoder Type | Speed | Memory | Biological Relevance | Stability |
|-------------|-------|--------|---------------------|-----------|
| Simple      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Efficient   | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Kmer        | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Motif       | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🔍 **Troubleshooting**

### If you still get indexing errors:
1. **Check genome file path**: Ensure the genome FASTA file exists and is readable
2. **Reduce sequence length**: Try smaller `max_seq_len` values
3. **Use simple encoder**: Start with `SimpleDNAEncoder` as it's most stable
4. **Check coordinates**: Ensure your DNA coordinates are valid

### If you get device errors:
1. **Check device placement**: Ensure all models are on the same device
2. **Use the test script**: Run `test_dna_encoders.py` to verify everything works
3. **Check CUDA memory**: Reduce batch size if you run out of memory

## 🎯 **Recommendations**

1. **Start with SimpleDNAEncoder**: It's the most stable and efficient
2. **Test before training**: Always run the test script first
3. **Monitor memory usage**: DNA encoders can be memory intensive
4. **Use validation**: The improved pipeline includes proper validation
5. **Save checkpoints**: The pipeline saves DNA encoder state in checkpoints

## 📝 **Key Changes Made**

1. **Fixed device handling** in all encoders
2. **Added SimpleDNAEncoder** as default option
3. **Fixed transformer integration** in main muse file
4. **Added proper error handling** and fallbacks
5. **Created comprehensive test suite**
6. **Updated training pipeline** with better defaults

The improved DNA encoders should now work without the indexing errors you were experiencing! 