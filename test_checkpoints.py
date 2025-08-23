#!/usr/bin/env python3
"""
Checkpoint Testing Script for Muse Training
Tests saved checkpoints and generates sample outputs
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
from tqdm import tqdm

# Import your training components
from muse_pipeline_improved import TrainingConfig, MuseTrainer, HiCDataset, custom_collate
from torch.utils.data import DataLoader

def load_checkpoint_for_testing(checkpoint_path, config):
    """Load a checkpoint and return the trainer with loaded state"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create trainer with config
    trainer = MuseTrainer(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model states
    trainer.maskgit_lowres.load_state_dict(checkpoint['maskgit_lowres_state'])
    trainer.maskgit_highres.load_state_dict(checkpoint['maskgit_highres_state'])
    
    # Load optimizer states
    trainer.optimizer_lowres.load_state_dict(checkpoint['optimizer_lowres_state'])
    trainer.optimizer_highres.load_state_dict(checkpoint['optimizer_highres_state'])
    
    # Load training state
    trainer.current_epoch = checkpoint.get('epoch', 0)
    trainer.global_step = checkpoint.get('global_step', 0)
    trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"Loaded checkpoint from epoch {trainer.current_epoch}")
    print(f"Global step: {trainer.global_step}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    return trainer

def test_checkpoint_generation(trainer, test_batch_size=4, num_samples=8):
    """Test generation capabilities of loaded checkpoint"""
    print("\n=== Testing Generation ===")
    
    # Set models to eval mode
    trainer.maskgit_lowres.eval()
    trainer.maskgit_highres.eval()
    
    # Get a test batch
    test_loader = DataLoader(
        trainer.train_dataset, 
        batch_size=test_batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=2
    )
    
    with torch.no_grad():
        for batch_idx, (lowres_batch, highres_batch, coords) in enumerate(test_loader):
            if batch_idx >= num_samples // test_batch_size:
                break
                
            lowres_batch = lowres_batch.to(trainer.device)
            highres_batch = highres_batch.to(trainer.device)
            coords = trainer.ensure_coord_tuples(coords)
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"Low-res shape: {lowres_batch.shape}")
            print(f"High-res shape: {highres_batch.shape}")
            print(f"Coordinates: {coords[:2]}...")  # Show first 2 coords
            
            # Test low-res generation
            try:
                with torch.amp.autocast('cuda', enabled=trainer.config.use_mixed_precision):
                    # Generate low-res
                    dummy_texts = [""] * len(coords)
                    lowres_generated = trainer.maskgit_lowres.generate(
                        texts=dummy_texts,
                        dna_coords=coords
                    )
                    
                    # Generate high-res using low-res as condition
                    highres_generated = trainer.maskgit_highres.generate(
                        texts=dummy_texts,
                        dna_coords=coords,
                        cond_images=lowres_batch  # Use real low-res as condition
                    )
                
                print(f"✓ Low-res generation successful: {lowres_generated.shape}")
                print(f"✓ High-res generation successful: {highres_generated.shape}")
                
                # Calculate some basic statistics
                lowres_stats = {
                    'min': lowres_generated.min().item(),
                    'max': lowres_generated.max().item(),
                    'mean': lowres_generated.mean().item(),
                    'std': lowres_generated.std().item()
                }
                
                highres_stats = {
                    'min': highres_generated.min().item(),
                    'max': highres_generated.max().item(),
                    'mean': highres_generated.mean().item(),
                    'std': highres_generated.std().item()
                }
                
                print(f"Low-res stats: {lowres_stats}")
                print(f"High-res stats: {highres_stats}")
                
                # Save sample images
                save_sample_images(
                    lowres_batch, highres_batch, 
                    lowres_generated, highres_generated,
                    coords, batch_idx
                )
                
            except Exception as e:
                print(f"✗ Generation failed: {e}")
                continue

def save_sample_images(lowres_real, highres_real, lowres_gen, highres_gen, coords, batch_idx):
    """Save sample images for visualization"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for i in range(min(4, lowres_real.size(0))):
        # Low-res real
        axes[0, i].imshow(lowres_real[i, 0].cpu().numpy(), cmap='Reds')
        axes[0, i].set_title(f'Low-res Real\n{coords[i]}')
        axes[0, i].axis('off')
        
        # Low-res generated
        axes[1, i].imshow(lowres_gen[i, 0].cpu().numpy(), cmap='Reds')
        axes[1, i].set_title('Low-res Generated')
        axes[1, i].axis('off')
        
        # High-res real
        axes[2, i].imshow(highres_real[i, 0].cpu().numpy(), cmap='Reds')
        axes[2, i].set_title('High-res Real')
        axes[2, i].axis('off')
        
        # High-res generated
        axes[3, i].imshow(highres_gen[i, 0].cpu().numpy(), cmap='Reds')
        axes[3, i].set_title('High-res Generated')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'checkpoint_test_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample images: checkpoint_test_batch_{batch_idx}.png")

def test_checkpoint_loss(trainer, num_batches=5):
    """Test loss calculation on a few batches"""
    print("\n=== Testing Loss Calculation ===")
    
    trainer.maskgit_lowres.train()
    trainer.maskgit_highres.train()
    
    test_loader = DataLoader(
        trainer.train_dataset,
        batch_size=4,  # Smaller batch for testing
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=2
    )
    
    total_losses = []
    
    for batch_idx, (lowres_batch, highres_batch, coords) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
            
        lowres_batch = lowres_batch.to(trainer.device)
        highres_batch = highres_batch.to(trainer.device)
        coords = trainer.ensure_coord_tuples(coords)
        
        try:
            with torch.cuda.amp.autocast(enabled=trainer.config.use_mixed_precision):
                # Calculate losses
                loss_lowres = trainer.maskgit_lowres(
                    lowres_batch,
                    dna_coords=coords,
                    train_only_generator=False
                )
                
                loss_highres = trainer.maskgit_highres(
                    highres_batch,
                    dna_coords=coords,
                    cond_images=lowres_batch,
                    train_only_generator=False
                )
                
                total_loss = loss_lowres + loss_highres
                total_losses.append(total_loss.item())
                
                print(f"Batch {batch_idx + 1}:")
                print(f"  Low-res loss: {loss_lowres.item():.4f}")
                print(f"  High-res loss: {loss_highres.item():.4f}")
                print(f"  Total loss: {total_loss.item():.4f}")
                
        except Exception as e:
            print(f"✗ Loss calculation failed for batch {batch_idx + 1}: {e}")
            continue
    
    if total_losses:
        avg_loss = np.mean(total_losses)
        print(f"\nAverage loss over {len(total_losses)} batches: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Test Muse checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpointbest.pt',
                       help='Checkpoint file to test')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Test specific epoch checkpoint (e.g., 12 for checkpoint_epoch_12.pt)')
    parser.add_argument('--test-generation', action='store_true',
                       help='Test generation capabilities')
    parser.add_argument('--test-loss', action='store_true',
                       help='Test loss calculation')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.epoch is not None:
        checkpoint_path = f"checkpoint_epoch_{args.epoch}.pt"
    else:
        checkpoint_path = args.checkpoint
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found!")
        print("Available checkpoints:")
        for file in os.listdir('.'):
            if file.startswith('checkpoint') and file.endswith('.pt'):
                print(f"  {file}")
        return
    
    # Create config
    config = TrainingConfig()
    
    # Load checkpoint
    try:
        trainer = load_checkpoint_for_testing(checkpoint_path, config)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Test generation
    if args.test_generation:
        test_checkpoint_generation(trainer, num_samples=args.num_samples)
    
    # Test loss calculation
    if args.test_loss:
        test_checkpoint_loss(trainer)
    
    # If no specific tests requested, run both
    if not args.test_generation and not args.test_loss:
        print("Running both generation and loss tests...")
        test_checkpoint_generation(trainer, num_samples=args.num_samples)
        test_checkpoint_loss(trainer)
    
    print("\n=== Checkpoint Testing Complete ===")

if __name__ == "__main__":
    main() 