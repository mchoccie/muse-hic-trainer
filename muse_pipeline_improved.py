# Updated muse_pipeline_improved.py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from pathlib import Path
import time
from tqdm import tqdm
import wandb  # for experiment tracking

from muse_maskgit_pytorch import MaskGit, MaskGitTransformer
from muse_maskgit_pytorch.vqgan_vae_hic_weighted import VQGanVAE
# Updated imports to use improved DNA encoders
from muse_maskgit_pytorch.improved_dna_encoders import create_dna_encoder

# ------------------------------------------------------------------
# 1) Enhanced Dataset with better preprocessing
# ------------------------------------------------------------------

class HiCDataset(Dataset):
    def __init__(self, lowres_np, highres_np, coords, augment=True):
        self.lowres_np = lowres_np
        self.highres_np = highres_np
        self.coords = coords
        self.augment = augment

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        lowres = torch.from_numpy(self.lowres_np[idx].copy()).float()
        highres = torch.from_numpy(self.highres_np[idx].copy()).float()
        coords = tuple(self.coords[idx])
        
        # Data augmentation for better generalization
        if self.augment and torch.rand(1) < 0.5:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                lowres = torch.flip(lowres, dims=[1])
                highres = torch.flip(highres, dims=[1])
            
            # Random vertical flip
            if torch.rand(1) < 0.5:
                lowres = torch.flip(lowres, dims=[2])
                highres = torch.flip(highres, dims=[2])
            
            # Random rotation (90, 180, 270 degrees)
            if torch.rand(1) < 0.5:
                k = torch.randint(1, 4, (1,)).item()
                lowres = torch.rot90(lowres, k, dims=[1, 2])
                highres = torch.rot90(highres, k, dims=[1, 2])
        
        return lowres, highres, coords

def custom_collate(batch):
    lowres, highres, coords = zip(*batch)

    fixed_coords = []
    for c in coords:
        # If c is ('chr1', array([0, 12800000])) or similar — flatten it
        if isinstance(c, (list, tuple, np.ndarray)):
            # Flatten inner list if nested (e.g., ['chr1', array([0, 12800000])])
            if len(c) == 3:
                flat = tuple(
                    x.item() if isinstance(x, np.generic) else x
                    for x in c
                )
                fixed_coords.append(flat)
            else:
                # Deeply nested case
                flat = []
                for x in c:
                    if isinstance(x, (list, tuple, np.ndarray)):
                        flat.extend(x.tolist() if hasattr(x, 'tolist') else x)
                    else:
                        flat.append(x)
                assert len(flat) == 3, f"Could not flatten coord: {c}"
                fixed_coords.append(tuple(flat))
        else:
            raise TypeError(f"Invalid coord type: {type(c)} in {c}")

    return (
        torch.stack(lowres),
        torch.stack(highres),
        fixed_coords,
    )

# ------------------------------------------------------------------
# 2) Training configuration
# ------------------------------------------------------------------

class TrainingConfig:
    def __init__(self):
        # Data paths
        self.hic_path_lowres = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/lowres_dataset.npy"
        self.hic_path_highres = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/highres_dataset.npy"
        self.coords_path = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/hic_window_coords.npy"
        
        # Model paths
        self.enformer_dir = '/scratch/rnd-rojas/Manan/enformer_local'
        self.genome_fasta = '/scratch/rnd-rojas/Manan/hg19.fa'
        self.vae_base_path = '/scratch/rnd-rojas/Manan/qv_results4/vae.best_srcc.pt'
        self.vae_highres_path = '/scratch/rnd-rojas/Manan/qv_results_highres_new/vae.best_srcc.pt'
        
        # DNA Encoder configuration - UPDATED to use simple encoder
        self.dna_encoder_type = 'simple'  # 'simple', 'efficient', 'kmer', 'motif'
        self.dna_embedding_dim = 256
        
        # Training hyperparameters
        self.batch_size = 8   # Reduced from 16 for memory
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4  # Added weight decay
        self.warmup_steps = 1000
        self.grad_clip_norm = 1.0
        
        # Model architecture
        self.transformer_dim = 512  # Reduced from 768 for memory
        self.transformer_depth = 8   # Reduced from 12 for memory
        self.transformer_heads = 8   # Reduced from 12 for memory
        self.transformer_dim_head = 64
        
        # Training strategy
        self.use_mixed_precision = False  # Disabled due to flash attention overflow
        self.gradient_accumulation_steps = 2
        self.save_every = 1000
        self.eval_every = 500
        self.log_every = 100
        
        # Loss weights
        self.critic_loss_weight = 0.5  # Reduced from 1.0
        
        # Validation split
        self.val_split = 0.1

# ------------------------------------------------------------------
# 3) Enhanced training loop
# ------------------------------------------------------------------

# --------------------------------------------------
# utility to guarantee List[Tuple[str,int,int]]
# --------------------------------------------------
def ensure_coord_tuples(coords):
    """Accept list/tuple/np.ndarray → return List[Tuple[str,int,int]]"""
    out = []
    for c in coords:
        if isinstance(c, tuple):
            out.append(c)
        elif isinstance(c, (list, np.ndarray)):
            # flatten and cast numpy scalars to Python ints
            flat = []
            for x in c:
                if isinstance(x, np.generic):
                    flat.append(int(x))
                else:
                    flat.append(x)
            if len(flat) != 3:
                raise ValueError(f"Bad coord (len {len(flat)}): {c}")
            out.append(tuple(flat))
        else:
            raise TypeError(f"Bad coord type {type(c)}: {c}")
    return out

class MuseTrainer:
    def __init__(self, config):
        self.config = config
        print(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self.writer = SummaryWriter('runs/muse_training')
        wandb.init(project="muse-hic", config=vars(config))
        
        # Load data
        self._load_data()
        
        # Initialize models
        self._initialize_models()
        
        # Initialize optimizers and schedulers
        self._initialize_optimizers()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _load_data(self):
        print("Loading data...")
        
        # Load numpy arrays
        self.hic_np_lowres = np.load(self.config.hic_path_lowres, mmap_mode="r")
        self.hic_np_highres = np.load(self.config.hic_path_highres, mmap_mode="r")
        coords = np.load(self.config.coords_path, allow_pickle=True)
        
        # 👈 ADD DEBUG PRINTS
        print(f"Raw coords type: {type(coords)}")
        print(f"Raw coords shape: {coords.shape}")
        print(f"Raw coords dtype: {coords.dtype}")
        print(f"First raw coord: {coords[0]}, type: {type(coords[0])}")
        
        self.coords = [tuple(c) for c in coords.tolist()]
        
        # 👈 ADD MORE DEBUG PRINTS
        print(f"After conversion - First coord: {self.coords[0]}, type: {type(self.coords[0])}")
        print(f"Sample of first 3 coords: {self.coords[:3]}")
        
        print(f"Data shapes: lowres {self.hic_np_lowres.shape}, highres {self.hic_np_highres.shape}")
        print(f"Number of samples: {len(self.coords)}")
        
        # Create train/val split
        num_val = int(len(self.coords) * self.config.val_split)
        num_train = len(self.coords) - num_val
        
        train_indices = list(range(num_train))
        val_indices = list(range(num_train, len(self.coords)))
        
        # Create datasets
        self.train_dataset = HiCDataset(
            self.hic_np_lowres[train_indices], 
            self.hic_np_highres[train_indices], 
            [self.coords[i] for i in train_indices],
            augment=True
        )
        
        self.val_dataset = HiCDataset(
            self.hic_np_lowres[val_indices], 
            self.hic_np_highres[val_indices], 
            [self.coords[i] for i in val_indices],
            augment=False
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=8,  # Increased from 4
            collate_fn=custom_collate,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=8,
            collate_fn=custom_collate,
            pin_memory=True
        )
        
    def _initialize_models(self):
        print("Initializing models...")
        
        # DNA encoder - UPDATED to use improved encoders with better error handling
        print(f"Creating {self.config.dna_encoder_type} DNA encoder...")
        try:
            self.dna_encoder = create_dna_encoder(
                encoder_type=self.config.dna_encoder_type,
                genome_fasta=self.config.genome_fasta,
                embedding_dim=self.config.dna_embedding_dim
            )
            print(f"Successfully created {self.config.dna_encoder_type} DNA encoder")
        except Exception as e:
            print(f"Failed to create {self.config.dna_encoder_type} encoder: {e}")
            print("Falling back to simple encoder...")
            self.dna_encoder = create_dna_encoder(
                encoder_type='simple',
                genome_fasta=self.config.genome_fasta,
                embedding_dim=self.config.dna_embedding_dim
            )
        
        # Move DNA encoder to device
        self.dna_encoder = self.dna_encoder.to(self.device)
        
        # VAEs
        self.vae_base = VQGanVAE(
            dim=256,
            codebook_size=1024,
            lookup_free_quantization = True,   # ← match checkpoint
            use_vgg_and_gan=False
        ).to(self.device)
        
        self.vae_highres = VQGanVAE(
            dim=512,
            codebook_size=4096,
            lookup_free_quantization = True,   # ← same here
            use_vgg_and_gan=False
        ).to(self.device)
        
        # Load pretrained VAEs
        self.vae_base.load(self.config.vae_base_path)
        self.vae_highres.load(self.config.vae_highres_path)
        
        # Transformers
        self.transformer_lowres = MaskGitTransformer(
            num_tokens=1024,
            dim=self.config.transformer_dim,
            seq_len=1024,
            depth=self.config.transformer_depth,
            heads=self.config.transformer_heads,
            dim_head=self.config.transformer_dim_head,
            dna_encoder=self.dna_encoder,  # Now using improved encoder
            self_cond=True,  # Enable self-conditioning
        ).to(self.device)
        
        self.transformer_highres = MaskGitTransformer(
            num_tokens=4096,  # Match the high-res VAE codebook size
            dim=self.config.transformer_dim,
            seq_len=1024,
            depth=self.config.transformer_depth,
            heads=self.config.transformer_heads,
            dim_head=self.config.transformer_dim_head,
            dna_encoder=self.dna_encoder,  # Now using improved encoder
            self_cond=True,
        ).to(self.device)
        
        # Note: Gradient checkpointing not available for this transformer implementation
        # Memory optimization is handled through mixed precision and smaller model size
        
        # MaskGit models
        self.maskgit_lowres = MaskGit(
            vae=self.vae_base,
            transformer=self.transformer_lowres,
            image_size=256,
            cond_drop_prob=0.1,  # Reduced from 0.5
            self_cond_prob=0.9,
            no_mask_token_prob=0.1,  # Added for better token diversity
        ).to(self.device)
        
        self.maskgit_highres = MaskGit(
            vae=self.vae_highres,
            transformer=self.transformer_highres,
            image_size=512,
            cond_image_size=256,
            cond_drop_prob=0.1,
            self_cond_prob=0.9,
            no_mask_token_prob=0.1,
        ).to(self.device)
        
    def _initialize_optimizers(self):
        # Optimizers with better hyperparameters
        self.optimizer_lowres = torch.optim.AdamW(
            self.maskgit_lowres.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.optimizer_highres = torch.optim.AdamW(
            self.maskgit_highres.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate schedulers
        self.scheduler_lowres = CosineAnnealingLR(
            self.optimizer_lowres, 
            T_max=self.config.num_epochs * len(self.train_loader),
            eta_min=1e-6
        )
        
        self.scheduler_highres = CosineAnnealingLR(
            self.optimizer_highres, 
            T_max=self.config.num_epochs * len(self.train_loader),
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_mixed_precision else None
        
    def train_epoch(self):
        self.maskgit_lowres.train()
        self.maskgit_highres.train()
        
        epoch_losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (lowres_batch, highres_batch, coords) in enumerate(pbar):
            # 👈 ADD DEBUG PRINTS FOR FIRST BATCH ONLY
            coords = ensure_coord_tuples(coords)
            if batch_idx == 0:
                print(f"Training loop - coords type: {type(coords)}")
                print(f"Training loop - coords length: {len(coords)}")
                print(f"Training loop - first coord: {coords[0]}, type: {type(coords[0])}")
                print(f"Training loop - first coord content: {coords[0]}")
                print(f"Training loop - sample coords: {coords[:3]}")
            
            # 👈 ADD VALIDATION HERE
            assert all(isinstance(c, tuple) and len(c) == 3 for c in coords), f"Bad coord batch: {coords}"
            
            lowres_batch = lowres_batch.to(self.device)
            highres_batch = highres_batch.to(self.device)
            
            # Gradient accumulation
            if batch_idx % self.config.gradient_accumulation_steps == 0:
                self.optimizer_lowres.zero_grad()
                self.optimizer_highres.zero_grad()
            
            # Mixed precision training
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Train low-res model
                    
                    loss_lowres = self.maskgit_lowres(
                        lowres_batch, 
                        dna_coords=coords,
                        train_only_generator=False
                    )
                    
                    # Train high-res model
                    loss_highres = self.maskgit_highres(
                        highres_batch, 
                        dna_coords=coords,
                        cond_images=lowres_batch,
                        train_only_generator=False
                    )
                    
                    total_loss = loss_lowres + loss_highres
                
                # Scale loss and backward pass
                scaled_loss = self.scaler.scale(total_loss / self.config.gradient_accumulation_steps)
                scaled_loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer_lowres)
                        self.scaler.unscale_(self.optimizer_highres)
                        torch.nn.utils.clip_grad_norm_(self.maskgit_lowres.parameters(), self.config.grad_clip_norm)
                        torch.nn.utils.clip_grad_norm_(self.maskgit_highres.parameters(), self.config.grad_clip_norm)
                    
                    self.scaler.step(self.optimizer_lowres)
                    self.scaler.step(self.optimizer_highres)
                    self.scaler.update()
                    
                    self.scheduler_lowres.step()
                    self.scheduler_highres.step()
            else:
                # Standard training
                loss_lowres = self.maskgit_lowres(
                    lowres_batch, 
                    dna_coords=coords,
                    train_only_generator=False
                )
                
                loss_highres = self.maskgit_highres(
                    highres_batch, 
                    dna_coords=coords,
                    cond_images=lowres_batch,
                    train_only_generator=False
                )
                
                total_loss = loss_lowres + loss_highres
                total_loss = total_loss / self.config.gradient_accumulation_steps
                total_loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.maskgit_lowres.parameters(), self.config.grad_clip_norm)
                        torch.nn.utils.clip_grad_norm_(self.maskgit_highres.parameters(), self.config.grad_clip_norm)
                    
                    self.optimizer_lowres.step()
                    self.optimizer_highres.step()
                    
                    self.scheduler_lowres.step()
                    self.scheduler_highres.step()
            
            epoch_losses.append(total_loss.item())
            self.global_step += 1
            
            # Memory management - clear cache every 10 batches
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Logging
            if self.global_step % self.config.log_every == 0:
                avg_loss = np.mean(epoch_losses[-self.config.log_every:])
                lr_lowres = self.optimizer_lowres.param_groups[0]['lr']
                lr_highres = self.optimizer_highres.param_groups[0]['lr']
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr_lowres': f'{lr_lowres:.2e}',
                    'lr_highres': f'{lr_highres:.2e}'
                })
                
                # Log to tensorboard and wandb
                self.writer.add_scalar('Loss/train', avg_loss, self.global_step)
                self.writer.add_scalar('LR/lowres', lr_lowres, self.global_step)
                self.writer.add_scalar('LR/highres', lr_highres, self.global_step)
                
                wandb.log({
                    'train_loss': avg_loss,
                    'lr_lowres': lr_lowres,
                    'lr_highres': lr_highres,
                    'epoch': self.current_epoch,
                    'step': self.global_step
                })
            
            # Save checkpoints
            if self.global_step % self.config.save_every == 0:
                self._save_checkpoint()
            
            # Validation
            if self.global_step % self.config.eval_every == 0:
                val_loss = self._validate()
                self.writer.add_scalar('Loss/val', val_loss, self.global_step)
                wandb.log({'val_loss': val_loss})
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best')
        
        return np.mean(epoch_losses)
    
    def _validate(self):
        self.maskgit_lowres.eval()
        self.maskgit_highres.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for lowres_batch, highres_batch, coords in self.val_loader:
                # 👈 ADD VALIDATION HERE
                coords = ensure_coord_tuples(coords)
                assert all(isinstance(c, tuple) and len(c) == 3 for c in coords), f"Bad coord batch in validation: {coords}"
                
                lowres_batch = lowres_batch.to(self.device)
                highres_batch = highres_batch.to(self.device)
                
                loss_lowres = self.maskgit_lowres(
                    lowres_batch, 
                    dna_coords=coords,
                    train_only_generator=True  # Only generator loss for validation
                )
                
                loss_highres = self.maskgit_highres(
                    highres_batch, 
                    dna_coords=coords,
                    cond_images=lowres_batch,
                    train_only_generator=True
                )
                
                total_loss = loss_lowres + loss_highres
                val_losses.append(total_loss.item())
        
        self.maskgit_lowres.train()
        self.maskgit_highres.train()
        
        return np.mean(val_losses)
    
    def _save_checkpoint(self, suffix=''):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'maskgit_lowres_state_dict': self.maskgit_lowres.state_dict(),
            'maskgit_highres_state_dict': self.maskgit_highres.state_dict(),
            'dna_encoder_state_dict': self.dna_encoder.state_dict(),  # Save DNA encoder state
            'optimizer_lowres_state_dict': self.optimizer_lowres.state_dict(),
            'optimizer_highres_state_dict': self.optimizer_highres.state_dict(),
            'scheduler_lowres_state_dict': self.scheduler_lowres.state_dict(),
            'scheduler_highres_state_dict': self.scheduler_highres.state_dict(),
            'config': self.config.__dict__
        }
        
        filename = f'checkpoint{suffix}.pt'
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.maskgit_lowres.load_state_dict(checkpoint['maskgit_lowres_state_dict'])
        self.maskgit_highres.load_state_dict(checkpoint['maskgit_highres_state_dict'])
        
        # Load DNA encoder state if available
        if 'dna_encoder_state_dict' in checkpoint:
            self.dna_encoder.load_state_dict(checkpoint['dna_encoder_state_dict'])
        
        self.optimizer_lowres.load_state_dict(checkpoint['optimizer_lowres_state_dict'])
        self.optimizer_highres.load_state_dict(checkpoint['optimizer_highres_state_dict'])
        self.scheduler_lowres.load_state_dict(checkpoint['scheduler_lowres_state_dict'])
        self.scheduler_highres.load_state_dict(checkpoint['scheduler_highres_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        print("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            epoch_loss = self.train_epoch()
            
            print(f"Epoch {epoch} completed. Average loss: {epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self._save_checkpoint(f'_epoch_{epoch}')
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        self.writer.close()
        wandb.finish()

# ------------------------------------------------------------------
# 4) Main training execution
# ------------------------------------------------------------------

if __name__ == "__main__":
    config = TrainingConfig()
    trainer = MuseTrainer(config)
    
    # Uncomment to resume from checkpoint
    # trainer.load_checkpoint('checkpoint.pt')
    
    trainer.train()