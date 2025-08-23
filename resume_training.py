#!/usr/bin/env python3
"""
Resume training from the last checkpoint
"""

from muse_pipeline_improved import TrainingConfig, MuseTrainer

def main():
    print("Resuming training from checkpoint...")
    
    # Create config
    config = TrainingConfig()
    
    # Create trainer
    trainer = MuseTrainer(config)
    
    # Load the latest checkpoint
    import os
    checkpoint_files = [f for f in os.listdir('.') if f.startswith('checkpoint') and f.endswith('.pt')]
    
    if checkpoint_files:
        # Find the latest epoch checkpoint
        epoch_checkpoints = [f for f in checkpoint_files if 'epoch' in f]
        if epoch_checkpoints:
            # Sort by epoch number
            epoch_checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            latest_checkpoint = epoch_checkpoints[-1]
            print(f"Loading latest checkpoint: {latest_checkpoint}")
            trainer.load_checkpoint(latest_checkpoint)
        else:
            # Use the best checkpoint
            print("Loading best checkpoint: checkpointbest.pt")
            trainer.load_checkpoint('checkpointbest.pt')
    else:
        print("No checkpoints found, starting from scratch")
    
    # Resume training
    trainer.train()

if __name__ == "__main__":
    main() 