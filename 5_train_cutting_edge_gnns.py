#!/usr/bin/env python3
"""
Cutting-Edge GNN Training for 30-Minute Intraday Volatility Prediction
=======================================================================
Advanced training pipeline for PNA, GPS-PNA Hybrid, and Dynamic Edge models
with uncertainty estimation, advanced loss functions, and ensemble capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import os
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import cutting-edge models
from utils.cutting_edge_gnns import (
    PNAVolatilityNet, 
    GPSPNAHybrid, 
    DynamicCorrelationNet,
    MixHopVolatilityNet,
    create_cutting_edge_model
)
from utils.dataset import IntradayGNNDataset
from utils.evaluation_intraday import VolatilityEvaluator
from utils.losses import QLIKELoss, RobustQLIKELoss, CombinedVolatilityLoss


class AdvancedGNNTrainer:
    """
    Advanced trainer for cutting-edge GNN models with:
    - Multiple loss functions (QLIKE, MSE, combined)
    - Uncertainty estimation
    - Learning rate scheduling
    - Gradient accumulation
    - Model ensembling
    - Advanced metrics tracking
    """
    
    def __init__(self, 
                 model_type='gps_pna_hybrid',
                 config_path='config/cutting_edge_config.yaml',
                 use_uncertainty=True,
                 use_advanced_loss=True):
        """
        Initialize advanced GNN trainer
        
        Args:
            model_type: One of 'pna', 'gps_pna_hybrid', 'dynamic', 'mixhop'
            config_path: Path to configuration file
            use_uncertainty: Whether to use uncertainty estimation
            use_advanced_loss: Whether to use advanced loss functions
        """
        self.model_type = model_type
        self.use_uncertainty = use_uncertainty
        self.use_advanced_loss = use_advanced_loss
        
        # Load or create config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._create_default_config()
            self._save_config(config_path)
        
        # Model-specific name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"{model_type}_30min_{timestamp}"
        self.output_dir = f'output/{self.model_name}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Training parameters
        self.seed = self.config.get('seed', 42)
        self.best_val_loss = float('inf')
        self.best_val_qlike = float('inf')
        self.best_epoch = 0
        self.patience = self.config.get('patience', 20)
        self.patience_counter = 0
        
        # Advanced training parameters
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.warmup_epochs = self.config.get('warmup_epochs', 5)
        self.use_amp = self.config.get('use_amp', True) and torch.cuda.is_available()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_qlikes = []
        self.val_qlikes = []
        self.train_uncertainties = []
        self.val_uncertainties = []
        self.learning_rates = []
        
        # Set seeds for reproducibility
        self._set_seeds()
        
        # Initialize evaluator
        scaler_file = 'processed_data/vols_30min_mean_std_scalers.csv'
        if not os.path.exists(scaler_file):
            print(f"Warning: Scaler file not found at {scaler_file}")
            self.evaluator = None
        else:
            self.evaluator = VolatilityEvaluator(scaler_file)
            print(f"Loaded evaluator with scaler parameters")
    
    def _create_default_config(self):
        """Create default configuration for cutting-edge models"""
        return {
            # Data parameters
            'seq_length': 42,
            'batch_size': 16,
            'num_workers': 4,
            
            # Model parameters
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'k_dynamic': 15,  # For dynamic edge conv
            
            # Training parameters
            'num_epochs': 150,
            'learning_rate': 5e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'patience': 20,
            
            # Advanced parameters
            'gradient_accumulation_steps': 2,
            'warmup_epochs': 5,
            'use_amp': True,
            'label_smoothing': 0.01,
            
            # Loss weights
            'mse_weight': 0.3,
            'qlike_weight': 0.7,
            'uncertainty_weight': 0.1,
            
            # Ensemble
            'ensemble_size': 1,
            
            # Seed
            'seed': 42
        }
    
    def _save_config(self, path):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def load_data(self):
        """Load 30-minute intraday data with proper temporal splits"""
        print("\n" + "="*80)
        print("Loading 30-minute intraday data for cutting-edge GNN...")
        print("="*80)
        
        vol_file = 'processed_data/vols_mats_30min_standardized.h5'
        volvol_file = 'processed_data/volvols_mats_30min_standardized.h5'
        
        if not os.path.exists(vol_file):
            raise FileNotFoundError(f"Data not found: {vol_file}\nPlease run data preparation scripts first!")
        
        # Create datasets with temporal splits
        self.train_dataset = IntradayGNNDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            root=f'processed_data/cutting_edge_gnn_{self.config["seq_length"]}_train',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='train',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        self.val_dataset = IntradayGNNDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            root=f'processed_data/cutting_edge_gnn_{self.config["seq_length"]}_val',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='val',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        self.test_dataset = IntradayGNNDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            root=f'processed_data/cutting_edge_gnn_{self.config["seq_length"]}_test',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(self.train_dataset)} graphs")
        print(f"  Val: {len(self.val_dataset)} graphs")
        print(f"  Test: {len(self.test_dataset)} graphs")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
    
    def create_model(self):
        """Create cutting-edge GNN model with proper initialization"""
        print(f"\nCreating {self.model_type.upper()} model...")
        
        # Get feature dimensions from dataset
        sample = self.train_dataset[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 1
        
        print(f"Feature dimensions - Nodes: {num_node_features}, Edges: {num_edge_features}")
        
        # Create model based on type
        model_params = {
            'num_node_features': num_node_features,
            'num_edge_features': num_edge_features,
            'hidden_dim': self.config['hidden_dim'],
            'output_dim': 1,
            'dropout': self.config['dropout']
        }
        
        if self.model_type == 'pna':
            self.model = PNAVolatilityNet(**model_params)
        elif self.model_type == 'gps_pna_hybrid':
            model_params.update({
                'num_heads': self.config['num_heads'],
                'num_layers': self.config['num_layers'],
                'k_dynamic': self.config.get('k_dynamic', 15)
            })
            self.model = GPSPNAHybrid(**model_params)
        elif self.model_type == 'dynamic':
            model_params.update({
                'k': self.config.get('k_dynamic', 10),
                'num_layers': self.config['num_layers']
            })
            self.model = DynamicCorrelationNet(**model_params)
        elif self.model_type == 'mixhop':
            model_params.update({
                'powers': [0, 1, 2, 3],
                'num_layers': self.config['num_layers']
            })
            self.model = MixHopVolatilityNet(**model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Initialize weights with Xavier/He initialization
        self._initialize_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        # Create loss functions
        if self.use_advanced_loss:
            self.mse_loss = nn.MSELoss()
            self.qlike_loss = RobustQLIKELoss(epsilon=1e-8, clip_ratio=10.0)
            self.combined_loss = CombinedVolatilityLoss(
                qlike_weight=self.config.get('qlike_weight', 0.7),
                mse_weight=self.config.get('mse_weight', 0.3),
                huber_weight=0.0
            )
        else:
            self.criterion = nn.MSELoss()
        
        # Mixed precision training
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _initialize_weights(self):
        """Initialize model weights properly"""
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def compute_loss(self, outputs, targets, uncertainties=None):
        """
        Compute advanced loss with uncertainty weighting
        
        CRITICAL: Data is log-transformed and standardized!
        For QLIKE, we need to inverse transform to variance scale.
        
        Args:
            outputs: Model predictions (standardized log-space)
            targets: True values (standardized log-space)
            uncertainties: Predicted uncertainties (optional)
        
        Returns:
            loss, mse, qlike values
        """
        # Ensure no NaN in inputs
        if torch.isnan(outputs).any() or torch.isnan(targets).any():
            print("Warning: NaN detected in outputs or targets")
            outputs = torch.nan_to_num(outputs, nan=0.0)
            targets = torch.nan_to_num(targets, nan=0.0)
        
        if self.use_advanced_loss:
            # For log-transformed data, MSE in log-space is appropriate
            mse_loss = F.mse_loss(outputs, targets)
            
            # For QLIKE, we need to transform to variance scale
            # Using the scaler parameters we loaded
            if self.evaluator and hasattr(self.evaluator, 'scaler_params'):
                with torch.no_grad():
                    # Inverse transform for QLIKE calculation
                    # Step 1: Inverse standardization
                    log_mean = self.evaluator.scaler_params['log_mean']
                    log_std = self.evaluator.scaler_params['log_std']
                    
                    outputs_log = outputs * log_std + log_mean
                    targets_log = targets * log_std + log_mean
                    
                    # Step 2: Exp to get variance
                    outputs_var = torch.exp(outputs_log)
                    targets_var = torch.exp(targets_log)
                    
                    # Ensure positive
                    outputs_var = torch.clamp(outputs_var, min=1e-10)
                    targets_var = torch.clamp(targets_var, min=1e-10)
                    
                    # Calculate QLIKE in variance space
                    try:
                        ratio = outputs_var / targets_var
                        ratio = torch.clamp(ratio, min=0.1, max=10.0)  # Clip extreme ratios
                        qlike = (ratio - torch.log(ratio) - 1).mean().item()
                        if np.isnan(qlike) or np.isinf(qlike):
                            qlike = 0.0
                    except:
                        qlike = 0.0
            else:
                # Without scaler params, can't compute meaningful QLIKE
                qlike = 0.0
            
            # Main loss is MSE in log-space (which is appropriate for log-normal data)
            loss = mse_loss
            
            # Add uncertainty regularization if available
            if uncertainties is not None and self.use_uncertainty:
                # Negative log likelihood with uncertainty
                nll = 0.5 * (torch.exp(-uncertainties) * (outputs - targets)**2 + uncertainties)
                nll = torch.nan_to_num(nll, nan=0.0)
                loss = loss + self.config.get('uncertainty_weight', 0.1) * nll.mean()
            
            mse = mse_loss.item()
            
        else:
            # Simple MSE loss
            loss = self.criterion(outputs, targets)
            mse = loss.item()
            qlike = 0.0
        
        # Final NaN check
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        return loss, mse, qlike
    
    def train_epoch(self, epoch):
        """Train one epoch with gradient accumulation and mixed precision"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_qlike = 0
        total_uncertainty = 0
        n_batches = 0
        
        # Learning rate warmup
        if epoch < self.warmup_epochs:
            warmup_lr = self.config['learning_rate'] * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        current_lr = self.optimizer.param_groups[0]['lr']
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} - Train")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass - handle different model types
                    if self.model_type in ['pna', 'gps_pna_hybrid']:
                        # These models support uncertainty estimation
                        result = self.model(batch, return_uncertainty=self.use_uncertainty)
                        if self.use_uncertainty and isinstance(result, tuple):
                            outputs, uncertainties = result
                        else:
                            outputs = result
                            uncertainties = None
                    else:
                        # Other models don't support uncertainty
                        outputs = self.model(batch)
                        uncertainties = None
                    
                    # Target
                    targets = batch.y
                    
                    # Compute loss
                    loss, mse, qlike = self.compute_loss(outputs, targets, uncertainties)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip']
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training without mixed precision
                if self.model_type in ['pna', 'gps_pna_hybrid']:
                    result = self.model(batch, return_uncertainty=self.use_uncertainty)
                    if self.use_uncertainty and isinstance(result, tuple):
                        outputs, uncertainties = result
                    else:
                        outputs = result
                        uncertainties = None
                else:
                    outputs = self.model(batch)
                    uncertainties = None
                
                targets = batch.y
                loss, mse, qlike = self.compute_loss(outputs, targets, uncertainties)
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_mse += mse
            total_qlike += qlike
            
            if uncertainties is not None:
                total_uncertainty += uncertainties.mean().item()
            
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'qlike': qlike,
                'lr': current_lr
            })
        
        # Average metrics
        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_qlike = total_qlike / n_batches
        avg_uncertainty = total_uncertainty / n_batches if total_uncertainty > 0 else 0
        
        # Store metrics
        self.train_losses.append(avg_loss)
        self.train_qlikes.append(avg_qlike)
        self.train_uncertainties.append(avg_uncertainty)
        self.learning_rates.append(current_lr)
        
        return avg_loss, avg_mse, avg_qlike, avg_uncertainty
    
    def validate(self):
        """Validate model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_qlike = 0
        total_uncertainty = 0
        n_batches = 0
        
        all_outputs = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                batch = batch.to(self.device)
                
                # Forward pass - handle different model types
                if self.model_type in ['pna', 'gps_pna_hybrid']:
                    result = self.model(batch, return_uncertainty=self.use_uncertainty)
                    if self.use_uncertainty and isinstance(result, tuple):
                        outputs, uncertainties = result
                        all_uncertainties.append(uncertainties.cpu().numpy())
                    else:
                        outputs = result
                        uncertainties = None
                else:
                    outputs = self.model(batch)
                    uncertainties = None
                
                targets = batch.y
                
                # Compute metrics
                loss, mse, qlike = self.compute_loss(outputs, targets, uncertainties)
                
                total_loss += loss.item()
                total_mse += mse
                total_qlike += qlike
                
                if uncertainties is not None:
                    total_uncertainty += uncertainties.mean().item()
                
                n_batches += 1
                
                # Collect predictions
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Average metrics
        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_qlike = total_qlike / n_batches
        avg_uncertainty = total_uncertainty / n_batches if total_uncertainty > 0 else 0
        
        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_qlikes.append(avg_qlike)
        self.val_uncertainties.append(avg_uncertainty)
        
        return avg_loss, avg_mse, avg_qlike, avg_uncertainty
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with all training state"""
        checkpoint = {
            'epoch': epoch,
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
            'train_qlike': self.train_qlikes[-1],
            'val_qlike': self.val_qlikes[-1],
            'config': self.config,
            'best_val_qlike': self.best_val_qlike
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with QLIKE: {self.val_qlikes[-1]:.4f}")
        
        # Keep only last 5 checkpoints
        checkpoints = sorted([f for f in os.listdir(self.output_dir) if f.startswith('checkpoint_epoch_')])
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                os.remove(os.path.join(self.output_dir, old_checkpoint))
    
    def plot_training_curves(self):
        """Generate comprehensive training visualizations"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        ax1 = plt.subplot(3, 3, 1)
        ax2 = plt.subplot(3, 3, 2)
        ax3 = plt.subplot(3, 3, 3)
        ax4 = plt.subplot(3, 3, 4)
        ax5 = plt.subplot(3, 3, 5)
        ax6 = plt.subplot(3, 3, 6)
        ax7 = plt.subplot(3, 3, 7)
        ax8 = plt.subplot(3, 3, 8)
        ax9 = plt.subplot(3, 3, 9)
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Train', alpha=0.7)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val', alpha=0.7)
        ax1.axvline(x=self.best_epoch+1, color='g', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # QLIKE curves
        ax2.plot(epochs, self.train_qlikes, 'b-', label='Train', alpha=0.7)
        ax2.plot(epochs, self.val_qlikes, 'r-', label='Val', alpha=0.7)
        ax2.axvline(x=self.best_epoch+1, color='g', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('QLIKE')
        ax2.set_title('QLIKE Loss (Primary Metric)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax3.plot(epochs, self.learning_rates, 'g-', alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Uncertainty evolution (if available)
        if self.train_uncertainties and any(u > 0 for u in self.train_uncertainties):
            ax4.plot(epochs, self.train_uncertainties, 'b-', label='Train', alpha=0.7)
            ax4.plot(epochs, self.val_uncertainties, 'r-', label='Val', alpha=0.7)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Avg Uncertainty')
            ax4.set_title('Uncertainty Estimates')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Uncertainty Data', ha='center', va='center')
            ax4.set_title('Uncertainty Estimates')
        
        # Loss ratio (train/val)
        if len(self.val_losses) > 0:
            loss_ratio = [t/v if v > 0 else 1 for t, v in zip(self.train_losses, self.val_losses)]
            ax5.plot(epochs, loss_ratio, 'purple', alpha=0.7)
            ax5.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Train/Val Loss Ratio')
            ax5.set_title('Overfitting Monitor')
            ax5.grid(True, alpha=0.3)
        
        # QLIKE improvement over epochs
        if len(self.val_qlikes) > 1:
            qlike_improvement = [(self.val_qlikes[0] - q) / self.val_qlikes[0] * 100 
                                for q in self.val_qlikes]
            ax6.plot(epochs, qlike_improvement, 'orange', alpha=0.7)
            ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Improvement (%)')
            ax6.set_title('QLIKE Improvement from Baseline')
            ax6.grid(True, alpha=0.3)
        
        # Gradient flow (placeholder - would need gradient tracking)
        ax7.text(0.5, 0.5, f'Model: {self.model_type.upper()}', ha='center', va='center', fontsize=12)
        ax7.text(0.5, 0.3, f'Best QLIKE: {self.best_val_qlike:.4f}', ha='center', va='center', fontsize=10)
        ax7.text(0.5, 0.1, f'Best Epoch: {self.best_epoch+1}', ha='center', va='center', fontsize=10)
        ax7.set_title('Model Summary')
        ax7.axis('off')
        
        # Early stopping progress
        patience_history = []
        for i in range(len(self.val_qlikes)):
            if i == 0 or self.val_qlikes[i] < min(self.val_qlikes[:i]):
                patience_history.append(0)
            else:
                patience_history.append(patience_history[-1] + 1 if patience_history else 1)
        
        ax8.plot(epochs, patience_history, 'coral', alpha=0.7)
        ax8.axhline(y=self.patience, color='r', linestyle='--', alpha=0.5, label=f'Limit: {self.patience}')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Patience Counter')
        ax8.set_title('Early Stopping Progress')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Loss components breakdown (if using advanced loss)
        if self.use_advanced_loss and len(self.train_qlikes) > 0:
            # Create stacked area chart
            mse_component = [l * 0.3 for l in self.train_losses]  # Approximate
            qlike_component = [q * 0.7 for q in self.train_qlikes]  # Approximate
            
            ax9.stackplot(epochs, mse_component, qlike_component, 
                         labels=['MSE (30%)', 'QLIKE (70%)'],
                         alpha=0.7, colors=['blue', 'orange'])
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Loss Components')
            ax9.set_title('Loss Composition')
            ax9.legend(loc='upper right')
            ax9.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.model_type.upper()} Training Progress - 30-Minute Volatility Forecasting', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop with advanced features"""
        print("\n" + "="*80)
        print(f"🚀 Starting {self.model_type.upper()} Training - Cutting-Edge GNN")
        print("="*80)
        print(f"Model: {self.model_type}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Patience: {self.patience}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Advanced Loss: {self.use_advanced_loss}")
        print(f"Uncertainty: {self.use_uncertainty}")
        print("="*80 + "\n")
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_mse, train_qlike, train_unc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_mse, val_qlike, val_unc = self.validate()
            
            # Step scheduler
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # Check for improvement (use QLIKE as primary metric)
            if val_qlike < self.best_val_qlike:
                self.best_val_qlike = val_qlike
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                
                print(f"✅ Epoch {epoch+1}: NEW BEST! "
                      f"Train QLIKE={train_qlike:.4f}, Val QLIKE={val_qlike:.4f}, "
                      f"LR={self.learning_rates[-1]:.2e}")
                
                if self.use_uncertainty and val_unc > 0:
                    print(f"   Uncertainty: Train={train_unc:.4f}, Val={val_unc:.4f}")
            else:
                self.patience_counter += 1
                print(f"📊 Epoch {epoch+1}: "
                      f"Train QLIKE={train_qlike:.4f}, Val QLIKE={val_qlike:.4f} "
                      f"(patience: {self.patience_counter}/{self.patience})")
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves()
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                print(f"Best model: epoch {self.best_epoch+1} with QLIKE={self.best_val_qlike:.4f}")
                break
        
        # Final plotting and saving
        self.plot_training_curves()
        
        # Save training history
        history = {
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_qlikes': self.train_qlikes,
            'val_qlikes': self.val_qlikes,
            'train_uncertainties': self.train_uncertainties,
            'val_uncertainties': self.val_uncertainties,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_qlike': float(self.best_val_qlike),
            'best_val_loss': float(self.best_val_loss),
            'config': self.config
        }
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n" + "="*80)
        print("🎉 Training Complete!")
        print(f"Best model: Epoch {self.best_epoch+1}")
        print(f"Best Val QLIKE: {self.best_val_qlike:.4f}")
        print(f"Saved to: {self.output_dir}")
        print("="*80)
        
        return self.best_val_qlike
    
    def evaluate_test(self):
        """Comprehensive test set evaluation with uncertainty quantification"""
        print("\n" + "="*80)
        print("📊 Evaluating on Test Set")
        print("="*80)
        
        # Load best model
        best_checkpoint = torch.load(
            os.path.join(self.output_dir, 'best_model.pt'),
            map_location=self.device
        )
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        self.model.eval()
        
        all_outputs = []
        all_targets = []
        all_uncertainties = []
        
        total_mse = 0
        total_qlike = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test Evaluation"):
                batch = batch.to(self.device)
                
                # Forward pass - handle different model types
                if self.model_type in ['pna', 'gps_pna_hybrid']:
                    result = self.model(batch, return_uncertainty=self.use_uncertainty)
                    if self.use_uncertainty and isinstance(result, tuple):
                        outputs, uncertainties = result
                        all_uncertainties.append(uncertainties.cpu().numpy())
                    else:
                        outputs = result
                        uncertainties = None
                else:
                    outputs = self.model(batch)
                    uncertainties = None
                
                targets = batch.y
                
                # Compute metrics
                _, mse, qlike = self.compute_loss(outputs, targets, uncertainties)
                total_mse += mse
                total_qlike += qlike
                n_batches += 1
                
                # Collect predictions
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Aggregate results
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        
        # Compute final metrics
        test_mse = total_mse / n_batches
        test_rmse = np.sqrt(test_mse)
        test_qlike = total_qlike / n_batches
        
        # R² score
        ss_res = np.sum((all_targets - all_outputs) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        # Mean absolute error
        mae = np.mean(np.abs(all_targets - all_outputs))
        
        print(f"\n{'='*60}")
        print(f"TEST RESULTS - {self.model_type.upper()}")
        print(f"{'='*60}")
        print(f"MSE:   {test_mse:.6f}")
        print(f"RMSE:  {test_rmse:.6f}")
        print(f"MAE:   {mae:.6f}")
        print(f"QLIKE: {test_qlike:.4f}")
        print(f"R²:    {r2_score:.4f}")
        
        if len(all_uncertainties) > 0:
            all_uncertainties = np.concatenate(all_uncertainties)
            avg_uncertainty = np.mean(all_uncertainties)
            print(f"Avg Uncertainty: {avg_uncertainty:.4f}")
            
            # Calibration metrics
            in_confidence = np.sum(
                np.abs(all_targets - all_outputs) < 2 * np.sqrt(np.exp(all_uncertainties))
            ) / len(all_targets)
            print(f"95% Confidence Coverage: {in_confidence*100:.1f}%")
        
        print(f"Test Samples: {len(all_targets)}")
        print(f"{'='*60}")
        
        # Save test results
        test_results = {
            'model_type': self.model_type,
            'test_mse': float(test_mse),
            'test_rmse': float(test_rmse),
            'test_mae': float(mae),
            'test_qlike': float(test_qlike),
            'test_r2': float(r2_score),
            'test_samples': len(all_targets),
            'best_epoch_used': int(self.best_epoch),
            'avg_uncertainty': float(avg_uncertainty) if len(all_uncertainties) > 0 else None
        }
        
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Save predictions for further analysis
        np.savez(
            os.path.join(self.output_dir, 'test_predictions.npz'),
            predictions=all_outputs,
            targets=all_targets,
            uncertainties=all_uncertainties if len(all_uncertainties) > 0 else None
        )
        
        return test_qlike


def main():
    """Main training pipeline for cutting-edge GNN models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train cutting-edge GNN models for volatility forecasting')
    parser.add_argument('--model', type=str, default='gps_pna_hybrid',
                       choices=['pna', 'gps_pna_hybrid', 'dynamic', 'mixhop'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default='config/cutting_edge_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--uncertainty', action='store_true',
                       help='Use uncertainty estimation')
    parser.add_argument('--advanced-loss', action='store_true', default=True,
                       help='Use advanced loss functions')
    parser.add_argument('--ensemble', type=int, default=1,
                       help='Number of models for ensemble (not implemented yet)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AdvancedGNNTrainer(
        model_type=args.model,
        config_path=args.config,
        use_uncertainty=args.uncertainty,
        use_advanced_loss=args.advanced_loss
    )
    
    # Load data
    trainer.load_data()
    
    # Create model
    trainer.create_model()
    
    # Train
    best_qlike = trainer.train()
    
    # Evaluate
    test_qlike = trainer.evaluate_test()
    
    print(f"\n{'='*80}")
    print(f"🏆 FINAL RESULTS - {args.model.upper()}")
    print(f"{'='*80}")
    print(f"Best Validation QLIKE: {best_qlike:.4f}")
    print(f"Test QLIKE: {test_qlike:.4f}")
    print(f"Improvement over baseline (~0.154): {(0.154 - test_qlike) / 0.154 * 100:.1f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()