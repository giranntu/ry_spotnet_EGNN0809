#!/usr/bin/env python3
"""
Training Script for Transformer Model - 30-Minute Volatility Forecasting
========================================================================
Professional training with proper metrics and evaluation
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import json
import yaml
from tqdm import tqdm
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.transformer_models import TransformerGNN
from utils.dataset import IntradayGNNDataset
from utils.evaluation_intraday import VolatilityEvaluator
from utils.losses import QLIKELoss


class TransformerTrainer:
    """Professional trainer for Transformer-based volatility forecasting"""
    
    def __init__(self, config_path='config/transformer_config.yaml'):
        """Initialize trainer with configuration"""
        
        # Default configuration
        self.config = {
            'modelname': 'TransformerGNN',
            'seq_length': 42,
            'batch_size': 16,
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'num_epochs': 100,
            'patience': 15,
            'gradient_clip': 0.5,
            'model': {
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'edge_dropout': 0.05,
                'use_layer_norm': True,
                'use_residual': True,
                'activation': 'gelu',
                'concat_heads': True,
                'beta': True
            }
        }
        
        # Load config if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self.config.update(loaded_config)
        else:
            # Save default config
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
        
        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Data paths
        self.vol_file = 'processed_data/vols_mats_30min_standardized.h5'
        self.volvol_file = 'processed_data/volvols_mats_30min_standardized.h5'
        self.scaler_file = 'processed_data/vols_30min_mean_std_scalers.csv'
        
        # Output directory
        self.output_dir = f'output/TransformerGNN_30min_{self.config["seq_length"]}'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        
        # Initialize components
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
    def setup_data(self):
        """Load datasets with proper train/val/test splits"""
        print("\nLoading datasets...")
        
        # Training dataset - 60% of data (2019-2022)
        self.train_dataset = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/transformer_gnn_{self.config["seq_length"]}_train',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='train',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        # Validation dataset - 20% of data (2023)
        self.val_dataset = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/transformer_gnn_{self.config["seq_length"]}_val',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='val',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        # Test dataset - 20% of data (2024-2025)
        self.test_dataset = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/transformer_gnn_{self.config["seq_length"]}_test',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        print(f"Data splits (30-minute intervals):")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        
        # Create data loaders
        self.train_loader = GeometricDataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        self.val_loader = GeometricDataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        self.test_loader = GeometricDataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Initialize evaluator
        self.evaluator = VolatilityEvaluator(self.scaler_file)
        
    def setup_model(self):
        """Initialize TransformerGNN model"""
        print("\nInitializing TransformerGNN...")
        
        # Get feature dimensions from sample
        sample = self.train_dataset[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
        
        print(f"Dataset dimensions: {num_node_features} node features, {num_edge_features} edge features")
        
        # Create model
        self.model = TransformerGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features if num_edge_features > 0 else 0,
            hidden_dim=self.config['model']['hidden_dim'],
            num_heads=self.config['model']['num_heads'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            edge_dropout=self.config['model']['edge_dropout'],
            use_layer_norm=self.config['model']['use_layer_norm'],
            use_residual=self.config['model']['use_residual'],
            activation=self.config['model']['activation'],
            output_dim=1,  # Predict 1 value per node
            concat_heads=self.config['model']['concat_heads'],
            beta=self.config['model']['beta']
        ).to(self.device)
        
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
    def setup_training(self):
        """Setup training components"""
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config['num_epochs']
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-7
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.qlike_loss = QLIKELoss()
        
        # Training history
        self.history = {
            'train_loss': [], 'train_mse': [], 'train_qlike': [],
            'val_loss': [], 'val_mse': [], 'val_qlike': [],
            'val_rmse': [], 'val_mae': []
        }
        
        # Best model tracking
        self.best_val_qlike = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        train_loss = 0
        train_mse = 0
        train_qlike = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            output = self.model(batch)
            target = batch.y.to(self.device)
            
            # Ensure proper shapes
            if output.shape != target.shape:
                if len(output.shape) == 2 and output.shape[1] == 1:
                    output = output.squeeze(1)
            
            # Calculate losses
            mse = self.mse_loss(output, target)
            
            # For QLIKE, ensure positive values
            output_exp = torch.exp(output)
            target_exp = torch.exp(target).clamp(min=0.01)
            qlike = self.qlike_loss(output_exp, target_exp)
            
            # Combined loss
            loss = 0.5 * mse + 0.5 * qlike
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clip']
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            train_loss += loss.item()
            train_mse += mse.item()
            train_qlike += qlike.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss / num_batches,
                'mse': train_mse / num_batches,
                'qlike': train_qlike / num_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return train_loss / num_batches, train_mse / num_batches, train_qlike / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        val_loss = 0
        val_mse = 0
        val_qlike = 0
        all_preds = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch = batch.to(self.device)
                
                output = self.model(batch)
                target = batch.y.to(self.device)
                
                # Ensure proper shapes
                if output.shape != target.shape:
                    if len(output.shape) == 2 and output.shape[1] == 1:
                        output = output.squeeze(1)
                
                # Calculate losses
                mse = self.mse_loss(output, target)
                
                output_exp = torch.exp(output)
                target_exp = torch.exp(target).clamp(min=0.01)
                qlike = self.qlike_loss(output_exp, target_exp)
                
                loss = 0.5 * mse + 0.5 * qlike
                
                val_loss += loss.item()
                val_mse += mse.item()
                val_qlike += qlike.item()
                num_batches += 1
                
                # Collect predictions for metrics
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Calculate additional metrics
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        
        # Reshape for evaluator if needed
        if y_pred.ndim == 1:
            n_samples = len(y_pred) // 30
            y_pred = y_pred.reshape(n_samples, 30)
            y_true = y_true.reshape(n_samples, 30)
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        avg_val_loss = val_loss / num_batches
        avg_val_mse = val_mse / num_batches
        avg_val_qlike = val_qlike / num_batches
        
        return avg_val_loss, avg_val_mse, avg_val_qlike, metrics
    
    def save_checkpoint(self, epoch, val_qlike, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_qlike': val_qlike,
            'config': self.config,
            'history': self.history
        }
        
        # Save regular checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                self.output_dir, 'checkpoints',
                f'checkpoint_epoch_{epoch:03d}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (QLIKE: {val_qlike:.4f})")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Training TransformerGNN for 30-Minute Volatility Forecasting")
        print("="*80)
        
        for epoch in range(self.config['num_epochs']):
            # Training
            train_loss, train_mse, train_qlike = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_mse, val_qlike, metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_mse'].append(train_mse)
            self.history['train_qlike'].append(train_qlike)
            self.history['val_loss'].append(val_loss)
            self.history['val_mse'].append(val_mse)
            self.history['val_qlike'].append(val_qlike)
            self.history['val_rmse'].append(metrics['rmse_vol'])
            self.history['val_mae'].append(metrics['mae_vol'])
            
            # Print results
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}:")
            print(f"  Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, QLIKE: {train_qlike:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, QLIKE: {val_qlike:.4f}")
            print(f"  Val   - RMSE: {metrics['rmse_vol']:.6f}, MAE: {metrics['mae_vol']:.6f}")
            
            # Check for improvement
            is_best = val_qlike < self.best_val_qlike
            if is_best:
                self.best_val_qlike = val_qlike
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_qlike, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final results
        self.save_final_results()
        
        print("\n" + "="*80)
        print("✅ Training Complete!")
        print(f"Best Validation QLIKE: {self.best_val_qlike:.4f}")
        print(f"Model saved to: {self.output_dir}")
        print("="*80)
    
    def save_final_results(self):
        """Save training history and final evaluation"""
        
        # Save history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.output_dir, 'training_history.csv'), index=False)
        
        # Save configuration
        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        self.model.eval()
        
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Test Evaluation'):
                batch = batch.to(self.device)
                output = self.model(batch)
                
                if len(output.shape) == 2 and output.shape[1] == 1:
                    output = output.squeeze(1)
                
                test_preds.append(output.cpu().numpy())
                test_targets.append(batch.y.cpu().numpy())
        
        y_pred = np.concatenate(test_preds)
        y_true = np.concatenate(test_targets)
        
        if y_pred.ndim == 1:
            n_samples = len(y_pred) // 30
            y_pred = y_pred.reshape(n_samples, 30)
            y_true = y_true.reshape(n_samples, 30)
        
        test_metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        # Save test results
        test_results = {
            'model': 'TransformerGNN',
            'test_qlike': float(test_metrics['qlike']),
            'test_rmse': float(test_metrics['rmse_vol']),
            'test_mae': float(test_metrics['mae_vol']),
            'test_mape': float(test_metrics['mape']),
            'best_val_qlike': float(self.best_val_qlike),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nTest Results:")
        print(f"  QLIKE: {test_metrics['qlike']:.4f}")
        print(f"  RMSE: {test_metrics['rmse_vol']:.6f}")
        print(f"  MAE: {test_metrics['mae_vol']:.6f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")


def main():
    """Main execution"""
    trainer = TransformerTrainer()
    trainer.train()


if __name__ == "__main__":
    main()