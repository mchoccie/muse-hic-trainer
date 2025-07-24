import optuna
import torch
import numpy as np
from muse_pipeline_improved import MuseTrainer, TrainingConfig
import json
from pathlib import Path

class HyperparameterOptimizer:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.best_params = None
        self.best_value = float('inf')
        
    def objective(self, trial):
        # Define hyperparameter search space
        config = TrainingConfig()
        
        # Learning rate
        config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        
        # Batch size
        config.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        # Transformer architecture
        config.transformer_dim = trial.suggest_categorical('transformer_dim', [512, 768, 1024])
        config.transformer_depth = trial.suggest_int('transformer_depth', 6, 16)
        config.transformer_heads = trial.suggest_int('transformer_heads', 8, 16)
        
        # Training parameters
        config.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        config.grad_clip_norm = trial.suggest_float('grad_clip_norm', 0.5, 2.0)
        config.critic_loss_weight = trial.suggest_float('critic_loss_weight', 0.1, 1.0)
        
        # Conditional dropout
        config.cond_drop_prob = trial.suggest_float('cond_drop_prob', 0.05, 0.3)
        
        # Number of epochs for quick evaluation
        config.num_epochs = 5  # Reduced for faster optimization
        
        try:
            # Initialize trainer with current config
            trainer = MuseTrainer(config)
            
            # Train for a few epochs to evaluate
            best_val_loss = float('inf')
            for epoch in range(config.num_epochs):
                trainer.current_epoch = epoch
                epoch_loss = trainer.train_epoch()
                val_loss = trainer._validate()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            return best_val_loss
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def optimize(self):
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(self.objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        print(f"Best validation loss: {self.best_value}")
        print(f"Best parameters: {self.best_params}")
        
        # Save results
        results = {
            'best_value': self.best_value,
            'best_params': self.best_params,
            'study_history': study.trials_dataframe().to_dict('records')
        }
        
        with open('hyperparameter_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return self.best_params

def run_hyperparameter_optimization():
    optimizer = HyperparameterOptimizer(n_trials=30)
    best_params = optimizer.optimize()
    
    # Create final config with best parameters
    final_config = TrainingConfig()
    for key, value in best_params.items():
        if hasattr(final_config, key):
            setattr(final_config, key, value)
    
    # Train final model with best parameters
    final_config.num_epochs = 100  # Full training
    trainer = MuseTrainer(final_config)
    trainer.train()

if __name__ == "__main__":
    run_hyperparameter_optimization() 