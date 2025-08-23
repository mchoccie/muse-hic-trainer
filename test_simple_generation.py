#!/usr/bin/env python3
"""
Simple test script to verify generation works
"""

import torch
import numpy as np
from muse_pipeline_improved import TrainingConfig, MuseTrainer

def test_simple_generation():
    """Test basic generation functionality"""
    print("Testing simple generation...")
    
    # Load model
    config = TrainingConfig()
    trainer = MuseTrainer(config)
    
    # Load checkpoint
    checkpoint_path = 'checkpointbest.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    trainer.maskgit_highres.load_state_dict(checkpoint['maskgit_highres_state_dict'])
    trainer.maskgit_highres.eval()
    
    print("✓ Model loaded successfully")
    
    # Create dummy data
    batch_size = 2
    lowres_image = torch.randn(batch_size, 1, 256, 256)
    dna_coords = [('chr1', 1000000, 2000000), ('chr2', 5000000, 6000000)]
    dummy_texts = [""] * batch_size
    
    print(f"Input shape: {lowres_image.shape}")
    print(f"DNA coords: {dna_coords}")
    
    # Test generation
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=False):
            try:
                highres_generated = trainer.maskgit_highres.generate(
                    texts=dummy_texts,
                    dna_coords=dna_coords,
                    cond_images=lowres_image
                )
                print(f"✓ Generation successful!")
                print(f"Output shape: {highres_generated.shape}")
                print(f"Output range: [{highres_generated.min():.3f}, {highres_generated.max():.3f}]")
                
            except Exception as e:
                print(f"✗ Generation failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_simple_generation() 