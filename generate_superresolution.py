#!/usr/bin/env python3
"""
Super-Resolution Generation Script
Takes low-res image + DNA sequence and generates high-res image
"""

import torch
import numpy as np
from muse_maskgit_pytorch.muse_maskgit_pytorch import generate_from_dna
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# Import your training components
from muse_pipeline_improved import TrainingConfig, MuseTrainer, HiCDataset, custom_collate
from torch.utils.data import DataLoader

def load_model_for_inference(checkpoint_path):
    """Load trained model for inference"""
    print(f"Loading model from: {checkpoint_path}")
    
    config = TrainingConfig()
    trainer = MuseTrainer(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    trainer.maskgit_highres.load_state_dict(checkpoint['maskgit_highres_state_dict'])
    
    # Set to eval mode
    trainer.maskgit_highres.eval()
    
    print("✓ Model loaded successfully")
    return trainer

def generate_highres_from_lowres(trainer, lowres_image, dna_coords, output_path=None):
    """
    Generate high-res image from low-res image + DNA coordinates
    
    Args:
        trainer: Loaded MuseTrainer with trained model
        lowres_image: torch.Tensor of shape [1, 256, 256] or [B, 1, 256, 256]
        dna_coords: List of tuples like [('chr1', 1000000, 2000000)]
        output_path: Optional path to save the generated image
    
    Returns:
        highres_generated: torch.Tensor of shape [B, 1, 512, 512]
    """
    device = trainer.device
    
    # Ensure proper shape
    if lowres_image.dim() == 3:
        lowres_image = lowres_image.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    lowres_image = lowres_image.to(device)
    
    # Ensure coords are in correct format
    if isinstance(dna_coords, str):
        # Parse coordinate string like "chr1:1000000-2000000"
        chrom, pos = dna_coords.split(':')
        start, end = map(int, pos.split('-'))
        dna_coords = [(chrom, start, end)]
    elif isinstance(dna_coords, tuple):
        dna_coords = [dna_coords]
    
    print(f"Input low-res shape: {lowres_image.shape}")
    print(f"DNA coordinates: {dna_coords}")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=trainer.config.use_mixed_precision):
            # Generate high-res using low-res as condition
            # highres_generated = trainer.maskgit_highres.generate(
            #     dna_coords=dna_coords,
            #     cond_images=lowres_image
            # )

            highres_generated = generate_from_dna(trainer.maskgit_highres,
                              dna_coords=dna_coords,
                              cond_images=lowres_image,
                              cond_scale=3.0)          # [B,1,512,512]
    
    print(f"✓ Generated high-res shape: {highres_generated.shape}")
    
    # Save if output path provided
    if output_path:
        save_generation_result(lowres_image, highres_generated, dna_coords, output_path)
    
    return highres_generated

def save_generation_result(lowres_image, highres_generated, dna_coords, output_path):
    """Save the generation result as an image"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Low-res input
    axes[0].imshow(lowres_image[0, 0].cpu().numpy(), cmap='Reds')
    axes[0].set_title(f'Input Low-Res\n{dna_coords[0]}')
    axes[0].axis('off')
    
    # High-res generated
    axes[1].imshow(highres_generated[0, 0].cpu().numpy(), cmap='Reds')
    axes[1].set_title('Generated High-Res')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved result to: {output_path}")

def generate_from_dataset_sample(trainer, checkpoint_path, sample_idx=0, output_path=None):
    """Generate from a sample in the dataset"""
    print(f"Generating from dataset sample {sample_idx}")
    
    # Load a sample from the dataset
    config = TrainingConfig()
    dataset = HiCDataset(
        np.load(config.hic_path_lowres, mmap_mode='r'),
        np.load(config.hic_path_highres, mmap_mode='r'),
        np.load(config.coords_path, allow_pickle=True).tolist()
    )
    
    # Get sample
    lowres_sample, highres_sample, coords_sample = dataset[sample_idx]
    lowres_sample = lowres_sample.unsqueeze(0)  # Add batch dimension
    
    print(f"Sample coordinates: {coords_sample}")
    
    # Generate
    highres_generated = generate_highres_from_lowres(
        trainer, lowres_sample, [coords_sample], output_path
    )
    
    return lowres_sample, highres_generated, coords_sample

def main():
    parser = argparse.ArgumentParser(description='Generate high-res from low-res + DNA')
    parser.add_argument('--checkpoint', type=str, default='checkpointbest.pt',
                       help='Checkpoint file to load')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Use specific epoch checkpoint')
    parser.add_argument('--lowres-path', type=str, default=None,
                       help='Path to low-res image (.npy or .png)')
    parser.add_argument('--dna-coords', type=str, default=None,
                       help='DNA coordinates (e.g., "chr1:1000000-2000000")')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Use dataset sample index (if no lowres-path provided)')
    parser.add_argument('--output', type=str, default='generated_highres.png',
                       help='Output path for generated image')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.epoch is not None:
        checkpoint_path = f"checkpoint_epoch_{args.epoch}.pt"
    else:
        checkpoint_path = args.checkpoint
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found!")
        return
    
    # Load model
    try:
        trainer = load_model_for_inference(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Generate based on input type
    if args.lowres_path and args.dna_coords:
        # Load custom low-res image
        print(f"Loading low-res image from: {args.lowres_path}")
        
        if args.lowres_path.endswith('.npy'):
            lowres_image = torch.from_numpy(np.load(args.lowres_path)).float()
        else:
            # Load as image and convert to tensor
            from PIL import Image
            img = Image.open(args.lowres_path).convert('L')  # Convert to grayscale
            lowres_image = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0
        
        # Generate
        highres_generated = generate_highres_from_lowres(
            trainer, lowres_image, args.dna_coords, args.output
        )
        
    else:
        # Use dataset sample
        lowres_sample, highres_generated, coords_sample = generate_from_dataset_sample(
            trainer, checkpoint_path, args.sample_idx, args.output
        )
    
    print("\n=== Generation Complete ===")
    print(f"High-res image shape: {highres_generated.shape}")
    print(f"Value range: [{highres_generated.min():.3f}, {highres_generated.max():.3f}]")

if __name__ == "__main__":
    main() 