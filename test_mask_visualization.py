#!/usr/bin/env python3
"""
Test script for mask visualization functionality
"""

import torch
import numpy as np
from muse_pipeline_improved import TrainingConfig, MuseTrainer, visualize_masked_tokens, get_masked_tokens_for_visualization

def test_mask_visualization():
    """Test the mask visualization functionality"""
    print("Testing mask visualization...")
    
    # Create config with visualization enabled
    config = TrainingConfig()
    config.visualize_mask_every = 1  # Visualize every step for testing
    config.save_mask_visualizations = True
    
    # Create trainer
    trainer = MuseTrainer(config)
    
    # Get a sample from the dataset
    sample_lo, sample_hi, sample_coords = trainer.train_ds[0]
    sample_hi = sample_hi.unsqueeze(0).to(trainer.device)  # Add batch dimension
    sample_lo = sample_lo.unsqueeze(0).to(trainer.device)  # Add batch dimension
    
    print(f"Sample shape: {sample_hi.shape}")
    print(f"Sample coords: {sample_coords}")
    
    # Test the masking visualization
    try:
        # Get masked tokens
        masked_tokens, original_tokens, mask = get_masked_tokens_for_visualization(
            trainer.maskgit_high, sample_hi, cond_images=sample_lo
        )
        
        print(f"Masked tokens shape: {masked_tokens.shape}")
        print(f"Mask ratio: {(masked_tokens != trainer.maskgit_high.mask_id).sum().item()}/{masked_tokens.shape[1]} tokens masked")
        
        # Visualize
        fig = visualize_masked_tokens(
            trainer.maskgit_high, sample_hi, masked_tokens, 0, save_dir="test_visualizations"
        )
        
        print("✓ Mask visualization test completed successfully!")
        
    except Exception as e:
        print(f"✗ Mask visualization test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mask_visualization()

