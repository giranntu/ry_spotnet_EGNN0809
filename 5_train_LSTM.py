#!/usr/bin/env python3
"""
LSTM Training for 30-Minute Intraday Volatility Prediction
===========================================================
Correctly uses 30-minute interval data with proper evaluation metrics
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Import our correct intraday dataset and evaluator
from utils.dataset import IntradayVolatilityDataset
from utils.evaluation_intraday import VolatilityEvaluator


class LSTMModel(nn.Module):
    """LSTM for 30-minute interval volatility prediction"""
    def __init__(self, input_size=930, hidden_size=256, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Output 30 values (volatilities for 30 stocks)
        self.fc = nn.Linear(hidden_size, 30)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last timestep for prediction
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMTrainer:
    def __init__(self, config_path='config/GNN_param.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Update config for 30-minute data
        self.config['seq_length'] = 42  # 42 thirty-minute intervals
        
        # Setup paths
        self.output_dir = f'output/LSTM_30min_{self.config["seq_length"]}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training parameters
        self.seed = self.config['seed'][0] if isinstance(self.config['seed'], list) else self.config['seed']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience = 15  # Early stopping patience
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_rmses = []
        self.val_rmses = []
        self.train_qlikes = []
        self.val_qlikes = []
        
        # Set seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize evaluator for proper metrics (REQUIRED)
        scaler_file = 'processed_data/vols_30min_mean_std_scalers.csv'
        if not os.path.exists(scaler_file):
            raise FileNotFoundError(
                f"CRITICAL: Scaler file not found at {scaler_file}. "
                f"Please run standardization (script 4) first!"
            )
        self.evaluator = VolatilityEvaluator(scaler_file)
        print(f"Loaded evaluator with scaler parameters from {scaler_file}")
            
    def prepare_data(self):
        """Load and prepare 30-minute intraday data"""
        print("Loading 30-minute intraday volatility data...")
        
        self.num_workers = min(4, os.cpu_count() or 1)
        
        # Use the correct 30-minute data files
        vol_file = 'processed_data/vols_mats_30min_standardized.h5'
        volvol_file = 'processed_data/volvols_mats_30min_standardized.h5'
        
        # Check if files exist
        if not os.path.exists(vol_file):
            raise FileNotFoundError(f"30-minute data not found: {vol_file}\nPlease run scripts 2 and 4 first!")
        
        # Create datasets using IntradayVolatilityDataset
        self.train_dataset = IntradayVolatilityDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='train',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        self.val_dataset = IntradayVolatilityDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='val',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        self.test_dataset = IntradayVolatilityDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        print(f"\n30-Minute Intraday Data Splits:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
    def create_model(self):
        """Initialize LSTM model with correct dimensions"""
        # Input: 930 features (30 vols + 435 covols + 30 volvols + 435 covolvols)
        # Output: 30 volatilities (diagonal elements)
        self.model = LSTMModel(
            input_size=930,
            hidden_size=256,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        # Use MSE for training in standardized space
        self.criterion = nn.MSELoss()
        
    def compute_metrics(self, outputs, targets):
        """Compute metrics in both standardized and original scales"""
        # Standardized scale MSE
        mse_std = self.criterion(outputs, targets).item()
        rmse_std = np.sqrt(mse_std)
        
        # If we have evaluator, compute real-scale metrics
        if self.evaluator:
            # Convert to numpy
            outputs_np = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            
            # Compute all metrics including QLIKE
            metrics = self.evaluator.calculate_all_metrics(outputs_np, targets_np, is_variance=True)
            rmse_real = metrics['rmse_vol']
            qlike = metrics['qlike']
        else:
            rmse_real = rmse_std
            qlike = 0.0
            
        return mse_std, rmse_std, rmse_real, qlike
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_qlike = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} - Train")
        for batch in pbar:
            features = batch['features'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute QLIKE for monitoring
            with torch.no_grad():
                _, _, _, qlike = self.compute_metrics(outputs, target)
                total_qlike += qlike
            
            n_batches += 1
            pbar.set_postfix({'loss': loss.item(), 'qlike': qlike})
        
        avg_loss = total_loss / n_batches
        avg_qlike = total_qlike / n_batches
        rmse = np.sqrt(avg_loss)
        
        self.train_losses.append(avg_loss)
        self.train_rmses.append(rmse)
        self.train_qlikes.append(avg_qlike)
        
        return avg_loss, rmse, avg_qlike
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_qlike = 0
        n_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                features = batch['features'].to(self.device)
                target = batch['target'].to(self.device)
                
                outputs = self.model(features)
                
                mse, rmse_std, rmse_real, qlike = self.compute_metrics(outputs, target)
                total_loss += mse
                total_qlike += qlike
                n_batches += 1
                
                all_outputs.append(outputs.cpu())
                all_targets.append(target.cpu())
        
        avg_loss = total_loss / n_batches
        avg_qlike = total_qlike / n_batches
        rmse = np.sqrt(avg_loss)
        
        self.val_losses.append(avg_loss)
        self.val_rmses.append(rmse)
        self.val_qlikes.append(avg_qlike)
        
        return avg_loss, rmse, avg_qlike
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
            'train_rmse': self.train_rmses[-1],
            'val_rmse': self.val_rmses[-1],
            'train_qlike': self.train_qlikes[-1] if self.train_qlikes else 0,
            'val_qlike': self.val_qlikes[-1] if self.val_qlikes else 0,
            'patience_counter': self.patience_counter
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            
        # Keep only last 5 checkpoints
        checkpoints = sorted([f for f in os.listdir(self.output_dir) if f.startswith('checkpoint_epoch_')])
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                os.remove(os.path.join(self.output_dir, old_checkpoint))
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', alpha=0.8)
        ax1.plot(self.val_losses, label='Val Loss', alpha=0.8)
        ax1.axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch {self.best_epoch+1}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RMSE curves
        ax2.plot(self.train_rmses, label='Train RMSE', alpha=0.8)
        ax2.plot(self.val_rmses, label='Val RMSE', alpha=0.8)
        ax2.axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch {self.best_epoch+1}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE (Standardized Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # QLIKE curves
        if self.train_qlikes:
            ax3.plot(self.train_qlikes, label='Train QLIKE', alpha=0.8)
            ax3.plot(self.val_qlikes, label='Val QLIKE', alpha=0.8)
            ax3.axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch {self.best_epoch+1}')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('QLIKE')
            ax3.set_title('QLIKE Loss (Variance Scale)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Early stopping progress
        patience_history = []
        for i in range(len(self.val_losses)):
            if i == 0 or self.val_losses[i] < min(self.val_losses[:i]):
                patience_history.append(0)
            else:
                patience_history.append(patience_history[-1] + 1 if patience_history else 1)
        
        ax4.plot(patience_history, label='Patience Counter', alpha=0.8, color='orange')
        ax4.axhline(y=self.patience, color='r', linestyle='--', alpha=0.5, label=f'Patience Limit ({self.patience})')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Patience')
        ax4.set_title('Early Stopping Progress')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=100)
        plt.close()
    
    def train(self):
        """Main training loop with early stopping"""
        print("="*80)
        print(f"Starting LSTM Training - 30-Minute Intraday Volatility Prediction")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Sequence length: {self.config['seq_length']} intervals (~{self.config['seq_length']/13:.1f} days)")
        print(f"Early stopping patience: {self.patience}")
        print("="*80)
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_rmse, train_qlike = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_rmse, val_qlike = self.validate()
            
            # Check for improvement (always use QLIKE for economic relevance)
            improvement_metric = val_qlike  # QLIKE is our primary metric
            
            if epoch == 0 or improvement_metric < self.best_val_loss:
                self.best_val_loss = improvement_metric
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"✅ Epoch {epoch+1}: New best! Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                      f"Val QLIKE={val_qlike:.4f}")
            else:
                self.patience_counter += 1
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                      f"Val QLIKE={val_qlike:.4f} (patience: {self.patience_counter}/{self.patience})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"Best model from epoch {self.best_epoch+1}")
                if self.val_qlikes:
                    print(f"Best validation QLIKE: {self.val_qlikes[self.best_epoch]:.4f}")
                break
            
            # Plot progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves()
        
        # Final plotting
        self.plot_training_curves()
        
        # Save training history
        history = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'train_rmses': [float(x) for x in self.train_rmses],
            'val_rmses': [float(x) for x in self.val_rmses],
            'train_qlikes': [float(x) for x in self.train_qlikes],
            'val_qlikes': [float(x) for x in self.val_qlikes],
            'best_epoch': int(self.best_epoch),
            'best_val_metric': float(self.best_val_loss),
            'early_stopped': bool(self.patience_counter >= self.patience)
        }
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*80)
        print("Training completed!")
        print(f"Best model saved at epoch {self.best_epoch+1}")
        if self.val_qlikes:
            print(f"Best validation QLIKE: {self.val_qlikes[self.best_epoch]:.4f}")
        print("="*80)
        
        return self.best_val_loss
    
    def evaluate_test(self):
        """Evaluate on test set with comprehensive metrics"""
        print("\nEvaluating on test set...")
        
        if len(self.test_loader) == 0:
            print("ERROR: Test loader is empty!")
            return None, None
        
        # Load best model
        best_checkpoint = torch.load(
            os.path.join(self.output_dir, 'best_model.pt'),
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        self.model.eval()
        total_loss = 0
        total_qlike = 0
        n_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test evaluation"):
                features = batch['features'].to(self.device)
                target = batch['target'].to(self.device)
                
                outputs = self.model(features)
                
                mse, rmse_std, rmse_real, qlike = self.compute_metrics(outputs, target)
                total_loss += mse
                total_qlike += qlike
                n_batches += 1
                
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        test_loss = total_loss / n_batches
        test_rmse = np.sqrt(test_loss)
        test_qlike = total_qlike / n_batches
        
        # Aggregate predictions for additional metrics
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        
        # Compute R² score
        ss_res = np.sum((all_targets - all_outputs) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        print(f"\n" + "="*60)
        print("TEST SET EVALUATION - 30-MINUTE INTRADAY")
        print("="*60)
        print(f"Test Loss (MSE): {test_loss:.6f}")
        print(f"Test RMSE (standardized): {test_rmse:.6f}")
        print(f"Test QLIKE: {test_qlike:.4f}")
        print(f"Test R² Score: {r2_score:.4f}")
        print(f"Test samples: {len(self.test_dataset)}")
        print("="*60)
        
        # Save test results
        test_results = {
            'test_loss': float(test_loss),
            'test_rmse': float(test_rmse),
            'test_qlike': float(test_qlike),
            'test_r2': float(r2_score),
            'test_samples': len(self.test_dataset),
            'best_epoch_used': int(self.best_epoch)
        }
        
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_loss, test_rmse


def main():
    trainer = LSTMTrainer()
    trainer.prepare_data()
    trainer.create_model()
    trainer.train()
    trainer.evaluate_test()


if __name__ == "__main__":
    main()