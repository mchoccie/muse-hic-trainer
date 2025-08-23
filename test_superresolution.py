#!/usr/bin/env python3
"""
Super-Resolution Testing Script for Muse Checkpoints
Specifically tests the super-resolution capabilities with proper metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr, spearmanr

# Import your training components
from muse_pipeline_improved import TrainingConfig, MuseTrainer, HiCDataset, custom_collate
from torch.utils.data import DataLoader

def calculate_sr_metrics(real_hr, generated_hr, real_lr, generated_lr):
    """Calculate super-resolution specific metrics"""
    metrics = {}
    
    # Convert to numpy for metric calculation
    real_hr_np = real_hr.cpu().numpy()
    generated_hr_np = generated_hr.cpu().numpy()
    real_lr_np = real_lr.cpu().numpy()
    generated_lr_np = generated_lr.cpu().numpy()
    
    # Calculate metrics for each sample in batch
    psnr_values = []
    ssim_values = []
    pearson_values = []
    spearman_values = []
    
    for i in range(real_hr_np.shape[0]):
        # High-res metrics
        hr_real = real_hr_np[i, 0]  # Remove channel dimension
        hr_gen = generated_hr_np[i, 0]
        
        # Low-res metrics  
        lr_real = real_lr_np[i, 0]
        lr_gen = generated_lr_np[i, 0]
        
        # PSNR
        psnr_hr = psnr(hr_real, hr_gen, data_range=hr_real.max() - hr_real.min())
        psnr_lr = psnr(lr_real, lr_gen, data_range=lr_real.max() - lr_real.min())
        psnr_values.append(psnr_hr)
        
        # SSIM
        ssim_hr = ssim(hr_real, hr_gen, data_range=hr_real.max() - hr_real.min())
        ssim_lr = ssim(lr_real, lr_gen, data_range=lr_real.max() - lr_real.min())
        ssim_values.append(ssim_hr)
        
        # Correlation coefficients
        pearson_hr, _ = pearsonr(hr_real.flatten(), hr_gen.flatten())
        spearman_hr, _ = spearmanr(hr_real.flatten(), hr_gen.flatten())
        pearson_values.append(pearson_hr)
        spearman_values.append(spearman_hr)
    
    metrics['psnr_mean'] = np.mean(psnr_values)
    metrics['psnr_std'] = np.std(psnr_values)
    metrics['ssim_mean'] = np.mean(ssim_values)
    metrics['ssim_std'] = np.std(ssim_values)
    metrics['pearson_mean'] = np.mean(pearson_values)
    metrics['pearson_std'] = np.std(pearson_values)
    metrics['spearman_mean'] = np.mean(spearman_values)
    metrics['spearman_std'] = np.std(spearman_values)
    
    return metrics

def test_superresolution_capabilities(trainer, num_samples=16):
    """Test super-resolution capabilities with comprehensive metrics"""
    print("\n=== Super-Resolution Testing ===")
    
    # Set models to eval mode
    trainer.maskgit_lowres.eval()
    trainer.maskgit_highres.eval()
    
    # Get test data
    test_loader = DataLoader(
        trainer.train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=2
    )
    
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, (lowres_batch, highres_batch, coords) in enumerate(test_loader):
            if batch_idx >= num_samples // 4:
                break
                
            lowres_batch = lowres_batch.to(trainer.device)
            highres_batch = highres_batch.to(trainer.device)
            coords = trainer.ensure_coord_tuples(coords)
            
            print(f"\n--- Batch {batch_idx + 1} ---")
            print(f"Coordinates: {coords[:2]}...")
            
            try:
                with torch.amp.autocast('cuda', enabled=trainer.config.use_mixed_precision):
                    # Method 1: Direct high-res generation from DNA
                    dummy_texts = [""] * len(coords)
                    highres_direct = trainer.maskgit_highres.generate(
                        texts=dummy_texts,
                        dna_coords=coords
                    )
                    
                    # Method 2: Two-stage generation (low-res -> high-res)
                    lowres_generated = trainer.maskgit_lowres.generate(
                        texts=dummy_texts,
                        dna_coords=coords
                    )
                    
                    highres_from_lowres = trainer.maskgit_highres.generate(
                        texts=dummy_texts,
                        dna_coords=coords,
                        cond_images=lowres_generated  # Use generated low-res
                    )
                    
                    # Method 3: Super-resolution using real low-res as condition
                    highres_sr = trainer.maskgit_highres.generate(
                        texts=dummy_texts,
                        dna_coords=coords,
                        cond_images=lowres_batch  # Use real low-res
                    )
                
                print("✓ All generation methods successful")
                
                # Calculate metrics for each method
                metrics_direct = calculate_sr_metrics(
                    highres_batch, highres_direct, lowres_batch, lowres_batch
                )
                
                metrics_two_stage = calculate_sr_metrics(
                    highres_batch, highres_from_lowres, lowres_batch, lowres_generated
                )
                
                metrics_sr = calculate_sr_metrics(
                    highres_batch, highres_sr, lowres_batch, lowres_batch
                )
                
                # Store results
                batch_results = {
                    'batch_idx': batch_idx,
                    'coords': coords,
                    'direct': metrics_direct,
                    'two_stage': metrics_two_stage,
                    'super_resolution': metrics_sr
                }
                all_metrics.append(batch_results)
                
                # Print results
                print("\nMetrics Summary:")
                print("Method 1 - Direct High-Res Generation:")
                print(f"  PSNR: {metrics_direct['psnr_mean']:.3f} ± {metrics_direct['psnr_std']:.3f}")
                print(f"  SSIM: {metrics_direct['ssim_mean']:.3f} ± {metrics_direct['ssim_std']:.3f}")
                print(f"  Pearson: {metrics_direct['pearson_mean']:.3f} ± {metrics_direct['pearson_std']:.3f}")
                
                print("\nMethod 2 - Two-Stage Generation:")
                print(f"  PSNR: {metrics_two_stage['psnr_mean']:.3f} ± {metrics_two_stage['psnr_std']:.3f}")
                print(f"  SSIM: {metrics_two_stage['ssim_mean']:.3f} ± {metrics_two_stage['ssim_std']:.3f}")
                print(f"  Pearson: {metrics_two_stage['pearson_mean']:.3f} ± {metrics_two_stage['pearson_std']:.3f}")
                
                print("\nMethod 3 - Super-Resolution:")
                print(f"  PSNR: {metrics_sr['psnr_mean']:.3f} ± {metrics_sr['psnr_std']:.3f}")
                print(f"  SSIM: {metrics_sr['ssim_mean']:.3f} ± {metrics_sr['ssim_std']:.3f}")
                print(f"  Pearson: {metrics_sr['pearson_mean']:.3f} ± {metrics_sr['pearson_std']:.3f}")
                
                # Save comparison images
                save_sr_comparison(
                    lowres_batch, highres_batch,
                    lowres_generated, highres_direct, highres_from_lowres, highres_sr,
                    coords, batch_idx
                )
                
            except Exception as e:
                print(f"✗ Generation failed: {e}")
                continue
    
    # Overall analysis
    if all_metrics:
        analyze_sr_results(all_metrics)

def save_sr_comparison(lowres_real, highres_real, lowres_gen, 
                       highres_direct, highres_two_stage, highres_sr, coords, batch_idx):
    """Save comprehensive super-resolution comparison images"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    for i in range(min(4, lowres_real.size(0))):
        # Row 1: Low-res comparison
        axes[0, i].imshow(lowres_real[i, 0].cpu().numpy(), cmap='Reds')
        axes[0, i].set_title(f'Real Low-Res\n{coords[i]}')
        axes[0, i].axis('off')
        
        # Row 2: Generated low-res
        axes[1, i].imshow(lowres_gen[i, 0].cpu().numpy(), cmap='Reds')
        axes[1, i].set_title('Generated Low-Res')
        axes[1, i].axis('off')
        
        # Row 3: High-res methods comparison
        axes[2, i].imshow(highres_real[i, 0].cpu().numpy(), cmap='Reds')
        axes[2, i].set_title('Real High-Res')
        axes[2, i].axis('off')
        
        # Row 4: Super-resolution result
        axes[3, i].imshow(highres_sr[i, 0].cpu().numpy(), cmap='Reds')
        axes[3, i].set_title('Super-Resolution')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sr_comparison_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create detailed comparison
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i in range(min(4, lowres_real.size(0))):
        # Method 1: Direct generation
        axes[0, i].imshow(highres_direct[i, 0].cpu().numpy(), cmap='Reds')
        axes[0, i].set_title('Direct High-Res')
        axes[0, i].axis('off')
        
        # Method 2: Two-stage
        axes[1, i].imshow(highres_two_stage[i, 0].cpu().numpy(), cmap='Reds')
        axes[1, i].set_title('Two-Stage')
        axes[1, i].axis('off')
        
        # Method 3: Super-resolution
        axes[2, i].imshow(highres_sr[i, 0].cpu().numpy(), cmap='Reds')
        axes[2, i].set_title('Super-Resolution')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sr_methods_comparison_batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SR comparison images for batch {batch_idx}")

def analyze_sr_results(all_metrics):
    """Analyze overall super-resolution results"""
    print("\n" + "="*80)
    print("SUPER-RESOLUTION ANALYSIS SUMMARY")
    print("="*80)
    
    # Aggregate metrics
    methods = ['direct', 'two_stage', 'super_resolution']
    method_names = ['Direct Generation', 'Two-Stage', 'Super-Resolution']
    
    for method, name in zip(methods, method_names):
        psnr_values = [m[method]['psnr_mean'] for m in all_metrics]
        ssim_values = [m[method]['ssim_mean'] for m in all_metrics]
        pearson_values = [m[method]['pearson_mean'] for m in all_metrics]
        
        print(f"\n{name}:")
        print(f"  PSNR: {np.mean(psnr_values):.3f} ± {np.std(psnr_values):.3f}")
        print(f"  SSIM: {np.mean(ssim_values):.3f} ± {np.std(ssim_values):.3f}")
        print(f"  Pearson: {np.mean(pearson_values):.3f} ± {np.std(pearson_values):.3f}")
    
    # Find best method
    avg_psnr = {}
    for method in methods:
        psnr_values = [m[method]['psnr_mean'] for m in all_metrics]
        avg_psnr[method] = np.mean(psnr_values)
    
    best_method = max(avg_psnr, key=avg_psnr.get)
    print(f"\nBest performing method: {method_names[methods.index(best_method)]}")
    print(f"Average PSNR: {avg_psnr[best_method]:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Test Muse super-resolution capabilities')
    parser.add_argument('--checkpoint', type=str, default='checkpointbest.pt',
                       help='Checkpoint file to test')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Test specific epoch checkpoint')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of samples to test')
    
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
    
    # Create config and load checkpoint
    config = TrainingConfig()
    
    try:
        from test_checkpoints import load_checkpoint_for_testing
        trainer = load_checkpoint_for_testing(checkpoint_path, config)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Test super-resolution capabilities
    test_superresolution_capabilities(trainer, num_samples=args.num_samples)
    
    print("\n=== Super-Resolution Testing Complete ===")

if __name__ == "__main__":
    main() 