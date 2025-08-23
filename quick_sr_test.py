#!/usr/bin/env python3
"""
Quick Super-Resolution Test
Minimal test to verify super-resolution generation works without DNA context
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Import your training components
from muse_pipeline_improved import TrainingConfig, MuseTrainer

def quick_test():
    """Quick test of super-resolution generation"""
    print("=== Quick Super-Resolution Test ===")
    
    # Create config
    config = TrainingConfig()
    print("✓ Config created")
    
    # Create trainer
    trainer = MuseTrainer(config)
    print("✓ Trainer created")
    
    # Set models to eval mode
    trainer.maskgit_low.eval()
    trainer.maskgit_high.eval()
    print("✓ Models set to eval mode")
    
    # Test generation with dummy data
    batch_size = 2
    dummy_texts = [""] * batch_size
    
    try:
        with torch.no_grad():
            # Test low-res generation
            print("\n--- Testing Low-Res Generation ---")
            lowres_generated = trainer.maskgit_low.generate(
                texts=dummy_texts
            )
            print(f"✓ Low-res generation: {lowres_generated.shape}")
            
            # Test high-res generation with conditioning
            print("\n--- Testing Super-Resolution ---")
            highres_sr = trainer.maskgit_high.generate(
                texts=dummy_texts,
                cond_images=lowres_generated
            )
            print(f"✓ Super-resolution: {highres_sr.shape}")
            
            # Test direct high-res generation
            print("\n--- Testing Direct High-Res Generation ---")
            highres_direct = trainer.maskgit_high.generate(
                texts=dummy_texts
            )
            print(f"✓ Direct high-res generation: {highres_direct.shape}")
            
            print("\n✓ All generation tests passed!")
            
            # Save a sample image
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            axes[0, 0].imshow(lowres_generated[0, 0].cpu().numpy(), cmap='Reds')
            axes[0, 0].set_title('Generated Low-Res')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(highres_sr[0, 0].cpu().numpy(), cmap='Reds')
            axes[0, 1].set_title('Super-Resolution')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(highres_direct[0, 0].cpu().numpy(), cmap='Reds')
            axes[1, 0].set_title('Direct High-Res')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(highres_sr[1, 0].cpu().numpy(), cmap='Reds')
            axes[1, 1].set_title('Super-Resolution (Sample 2)')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig('quick_sr_test.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✓ Sample image saved as 'quick_sr_test.png'")
            
            return True
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n=== Quick Test Complete ===")
        print("Your super-resolution setup is working correctly!")
        print("You can now run the full test scripts:")
        print("  python test_sr_simple.py --checkpoint your_checkpoint.pt")
        print("  python test_superresolution.py --checkpoint your_checkpoint.pt")
    else:
        print("\n=== Test Failed ===")
        print("There might be an issue with your setup.")


