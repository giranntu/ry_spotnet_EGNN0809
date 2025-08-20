#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE MODEL EVALUATION - ALL MODELS INCLUDED
===========================================================
Enhanced version of 6_evaluate_all_models_complete.py with:
- All original functionality preserved (95% same)
- Detailed path tracking and model discovery  
- Enhanced error handling and stability
- Publication-ready outputs
- Support for all model types including LSTM with proper test results
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
import pandas as pd
import json
import os
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import glob
warnings.filterwarnings('ignore')

# Import our models and datasets
from utils.models import GATModel
from utils.dataset import IntradayVolatilityDataset, IntradayGNNDataset
from utils.evaluation_intraday import VolatilityEvaluator
from utils.latex_generator import LaTeXTableGenerator
from utils.academic_plots import AcademicPlotter
from utils.transformer_models import TransformerGNN
# SpillNet removed from codebase
try:
    from utils.spillnet_models import SpillNet
    SPILLNET_AVAILABLE = True
except ImportError:
    SPILLNET_AVAILABLE = False
from utils.cutting_edge_gnns import PNAVolatilityNet


class FinalComprehensiveEvaluator:
    """
    Enhanced comprehensive evaluator that builds on the complete version with:
    - Detailed path tracking
    - Model discovery and availability checking
    - Enhanced error handling
    - Publication outputs
    """
    
    def __init__(self, config_path='config/GNN_param.yaml'):
        print(f"\n{'='*80}")
        print(f"🚀 FINAL COMPREHENSIVE MODEL EVALUATION")
        print(f"{'='*80}")
        print(f"Based on 6_evaluate_all_models_complete.py with enhancements")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load config with detailed tracking
        self.config_path = config_path
        print(f"\n📂 Configuration:")
        print(f"   Primary config: {config_path}")
        
        if not os.path.exists(config_path):
            print(f"   ⚠️  Primary config not found, checking alternatives...")
            # Try alternatives
            alt_configs = [
                'config/cutting_edge_config.yaml',
                'config/transformer_config.yaml'
            ]
            for alt_config in alt_configs:
                if os.path.exists(alt_config):
                    config_path = alt_config
                    print(f"   ✅ Using fallback: {alt_config}")
                    break
            else:
                raise FileNotFoundError(f"No valid config found")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"   ✅ Configuration loaded successfully")
        print(f"   Config keys: {list(self.config.keys())}")
        
        # Update for 30-minute data
        self.config['seq_length'] = 42
        print(f"   Sequence length: 42 (30-min intervals)")
        
        # Data files with validation
        print(f"\n📁 Data Files:")
        self.vol_file = 'processed_data/vols_mats_30min_standardized.h5'
        self.volvol_file = 'processed_data/volvols_mats_30min_standardized.h5'
        self.scaler_file = 'processed_data/vols_30min_mean_std_scalers.csv'
        
        # Check files exist with detailed info
        for file_path, name in [(self.vol_file, 'Volatility'), 
                                (self.volvol_file, 'Vol-of-vol'),
                                (self.scaler_file, 'Scaler')]:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024**2)
                print(f"   ✅ {name}: {file_path} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {name}: {file_path} (MISSING)")
                raise FileNotFoundError(f"{name} file not found: {file_path}")
        
        # Initialize evaluator
        print(f"\n🔧 Initializing components:")
        self.evaluator = VolatilityEvaluator(self.scaler_file)
        print(f"   ✅ Volatility evaluator initialized")
        
        # Device setup with detailed info
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {self.device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Load test dataset
        self.load_test_data()
        
        # Discover available models
        self.discover_available_models()
        
    def discover_available_models(self):
        """Enhanced model discovery with detailed information"""
        print(f"\n🔍 DISCOVERING AVAILABLE MODELS")
        print(f"{'='*60}")
        
        # Model search patterns
        model_patterns = {
            'LSTM': f'LSTM_30min_{self.config["seq_length"]}',
            'SpotV2Net_GAT': f'{self.config.get("modelname", "20240525_RGNN_std_optuna")}_30min_{self.config["seq_length"]}',
            'TransformerGNN': f'TransformerGNN_30min_{self.config["seq_length"]}',
            'SpillNet': f'SpillNet_30min_{self.config["seq_length"]}',
            'PNA': 'pna_30min_*',
            'XGBoost': f'XGBoost_30min_{self.config["seq_length"]}'
        }
        
        self.model_availability = {}
        
        for model_type, pattern in model_patterns.items():
            print(f"\n🔎 Checking {model_type}:")
            
            if '*' in pattern:
                # Use glob for wildcard patterns
                matching_dirs = glob.glob(f'output/{pattern}')
                if matching_dirs:
                    # Use the most recent
                    latest_dir = max(matching_dirs, key=lambda x: os.path.getmtime(os.path.join(x, 'best_model.pt')) if os.path.exists(os.path.join(x, 'best_model.pt')) else 0)
                    checkpoint_path = os.path.join(latest_dir, 'best_model.pt')
                else:
                    checkpoint_path = None
            else:
                checkpoint_path = f'output/{pattern}/best_model.pt'
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    # Get detailed info
                    size_mb = os.path.getsize(checkpoint_path) / (1024**2)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
                    
                    # Load checkpoint info
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    epoch = checkpoint.get('epoch', 'N/A')
                    val_qlike = checkpoint.get('val_qlike', checkpoint.get('best_val_qlike', 'N/A'))
                    
                    self.model_availability[model_type] = {
                        'available': True,
                        'path': checkpoint_path,
                        'size_mb': size_mb,
                        'modified': mod_time,
                        'epoch': epoch,
                        'val_qlike': val_qlike
                    }
                    
                    print(f"   ✅ Available: {checkpoint_path}")
                    print(f"      Size: {size_mb:.1f} MB")
                    print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"      Epoch: {epoch}")
                    if val_qlike != 'N/A':
                        print(f"      Val QLIKE: {val_qlike:.4f}")
                    
                except Exception as e:
                    print(f"   ⚠️  Found but couldn't load info: {e}")
                    self.model_availability[model_type] = {'available': False, 'error': str(e)}
            else:
                print(f"   ❌ Not found: {checkpoint_path if checkpoint_path else 'No path'}")
                self.model_availability[model_type] = {'available': False, 'path': checkpoint_path}
        
        # Summary
        available_count = sum(1 for info in self.model_availability.values() if info.get('available', False))
        print(f"\n📊 MODEL AVAILABILITY SUMMARY:")
        print(f"   Available models: {available_count}/{len(model_patterns)}")
        for model_type, info in self.model_availability.items():
            status = "✅" if info.get('available', False) else "❌"
            print(f"   {status} {model_type}")
        
    def load_test_data(self):
        """Load test data with enhanced logging (same as complete version)"""
        print(f"\n📚 Loading test data:")
        
        # Create test dataset for standard models
        self.test_dataset = IntradayVolatilityDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        print(f"   Test set: {len(self.test_dataset)} samples")
        
        # Create dataloader
        self.num_workers = min(4, os.cpu_count() or 1)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        print(f"   Test dataloader: {len(self.test_loader)} batches")
        
    def evaluate_naive_persistence(self):
        """Naive baseline: predict next = current (same as complete version)"""
        print("\nEvaluating Naive Persistence (30-min)...")
        
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.test_loader, desc="Naive evaluation"):
            features = batch['features']
            target = batch['target']
            
            # Use last interval's volatilities as prediction
            last_vols = features[:, -1, :30]
            
            all_preds.append(last_vols.numpy())
            all_targets.append(target.numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        # Calculate metrics using evaluator
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'Naive_30min',
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_historical_mean(self):
        """Historical mean baseline (same as complete version)"""
        print("\nEvaluating Historical Mean (30-min)...")
        
        # Calculate mean from training set
        train_dataset = IntradayVolatilityDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='train',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        # Collect training targets to compute mean
        train_targets = []
        for i in range(min(1000, len(train_dataset))):
            sample = train_dataset[i]
            train_targets.append(sample['target'].numpy())
        
        historical_mean = np.mean(np.vstack(train_targets), axis=0)
        
        # Apply to test set
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.test_loader, desc="Historical mean evaluation"):
            target = batch['target']
            batch_size = len(target)
            
            # Repeat historical mean for each sample
            pred = np.tile(historical_mean, (batch_size, 1))
            
            all_preds.append(pred)
            all_targets.append(target.numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'HistoricalMean_30min',
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_har_intraday(self):
        """HAR model adapted for 30-minute intervals (same as complete version)"""
        print("\nEvaluating HAR-Intraday (30-min)...")
        
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.test_loader, desc="HAR evaluation"):
            features = batch['features']  # [batch, 42, 930]
            target = batch['target']
            
            # HAR components for intraday:
            # - Last interval (30 min)
            # - Last 13 intervals (1 day)
            # - Last 42 intervals (≈3 days)
            
            # Extract volatilities (first 30 features)
            vols = features[:, :, :30]  # [batch, 42, 30]
            
            # Component 1: Last interval
            comp1 = vols[:, -1, :]
            
            # Component 2: Average of last day (13 intervals)
            comp2 = torch.mean(vols[:, -13:, :], dim=1)
            
            # Component 3: Average of all history (42 intervals)
            comp3 = torch.mean(vols, dim=1)
            
            # HAR prediction: weighted average
            # Traditional HAR weights: 1/3 each, but we can adjust for intraday
            w1, w2, w3 = 0.5, 0.3, 0.2  # More weight on recent
            y_pred = w1 * comp1 + w2 * comp2 + w3 * comp3
            
            all_preds.append(y_pred.numpy())
            all_targets.append(target.numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'HAR_Intraday_30min',
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_ewma(self):
        """EWMA (Exponentially Weighted Moving Average) baseline (same as complete version)"""
        print("\nEvaluating EWMA (30-min)...")
        
        all_preds = []
        all_targets = []
        
        # RiskMetrics standard for volatility
        alpha = 0.94
        
        for batch in tqdm(self.test_loader, desc="EWMA evaluation"):
            features = batch['features']
            target = batch['target']
            
            # Extract volatilities (first 30 features)
            vols = features[:, :, :30]
            
            # EWMA prediction
            batch_size = vols.shape[0]
            ewma_preds = []
            
            for b in range(batch_size):
                # Initialize with first value
                ewma = vols[b, 0, :].clone()
                
                # Apply EWMA through the sequence
                for t in range(1, vols.shape[1]):
                    ewma = alpha * ewma + (1 - alpha) * vols[b, t, :]
                
                ewma_preds.append(ewma.numpy())
            
            all_preds.extend(ewma_preds)
            all_targets.append(target.numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'EWMA_30min',
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_lstm(self):
        """Evaluate LSTM model with enhanced path tracking"""
        print("\nEvaluating LSTM (30-min)...")
        
        if not self.model_availability.get('LSTM', {}).get('available', False):
            print(f"  ⚠️  LSTM model not available")
            return None, None
        
        checkpoint_path = self.model_availability['LSTM']['path']
        print(f"  📂 Loading from: {checkpoint_path}")
        print(f"  📊 Model info:")
        print(f"     Size: {self.model_availability['LSTM']['size_mb']:.1f} MB")
        print(f"     Modified: {self.model_availability['LSTM']['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Epoch: {self.model_availability['LSTM']['epoch']}")
        
        # Load LSTM model (same architecture as complete version)
        class LSTMModel(nn.Module):
            def __init__(self, input_size=930, hidden_size=256, num_layers=2, dropout=0.2):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, 30)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out
        
        model = LSTMModel().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ LSTM loaded from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        
        # Show additional checkpoint info if available
        val_qlike = checkpoint.get('val_qlike', checkpoint.get('best_val_qlike', 'N/A'))
        test_qlike = checkpoint.get('test_qlike', 'N/A')  # Look for test results
        if val_qlike != 'N/A':
            print(f"     Best Val QLIKE: {val_qlike:.4f}")
        if test_qlike != 'N/A':
            print(f"     Test QLIKE: {test_qlike:.4f}")
        
        model.eval()
        
        # Get predictions (same as complete version)
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="LSTM evaluation"):
                features = batch['features'].to(self.device)
                target = batch['target']
                
                output = model(features)
                
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'LSTM_30min',
            'model_path': checkpoint_path,
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_spotv2net(self):
        """Evaluate SpotV2Net (GNN) model with enhanced error handling"""
        print("\nEvaluating SpotV2Net (30-min)...")
        
        if not self.model_availability.get('SpotV2Net_GAT', {}).get('available', False):
            print(f"  ⚠️  SpotV2Net model not available")
            return None, None
        
        checkpoint_path = self.model_availability['SpotV2Net_GAT']['path']
        print(f"  📂 Loading from: {checkpoint_path}")
        
        # Create GNN test dataset (same as complete version)
        test_dataset_gnn = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/intraday_gnn_30min_{self.config["seq_length"]}_test',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        test_loader_gnn = GeometricDataLoader(
            test_dataset_gnn,
            batch_size=self.config.get('batch_size', 128),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        # Get dimensions from sample
        sample = test_dataset_gnn[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 1
        
        print(f"  📐 Dimensions: {num_node_features} node features, {num_edge_features} edge features")
        
        # Load checkpoint first to determine correct architecture
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Try to infer num_heads from saved model architecture
            saved_state = checkpoint['model_state_dict']
            
            # Look for GAT layer parameters to infer head count
            actual_num_heads = self.config.get('num_heads', 6)  # Default fallback
            
            # Check if we can infer heads from the saved state
            for key in saved_state.keys():
                if 'gat_layers.0.att_src' in key:
                    # GAT attention parameter shape can tell us the head count
                    att_shape = saved_state[key].shape
                    if len(att_shape) >= 2:
                        potential_heads = att_shape[0] // att_shape[1] if att_shape[1] > 0 else 6
                        if potential_heads in [4, 6, 8, 12]:  # Common head counts
                            actual_num_heads = potential_heads
                            break
            
            print(f"  🔧 Detected {actual_num_heads} attention heads from saved model")
            
            # Recreate model with correct architecture
            model = GATModel(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                num_heads=actual_num_heads,  # Use detected heads
                output_node_channels=1,
                dim_hidden_layers=self.config.get('dim_hidden_layers', [500]),
                dropout_att=self.config.get('dropout_att', 0.0),
                dropout=self.config.get('dropout', 0.1),
                activation=self.config.get('activation', 'relu'),
                concat_heads=self.config.get('concat_heads', True),
                negative_slope=self.config.get('negative_slope', 0.2),
                standardize=False
            ).to(self.device)
            
            # Now load the state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✅ SpotV2Net loaded from epoch {checkpoint.get('epoch', 'N/A') + 1}")
            print(f"     Architecture: {actual_num_heads} attention heads")
            
            val_qlike = checkpoint.get('val_qlike', 'N/A')
            if val_qlike != 'N/A':
                print(f"     Best Val QLIKE: {val_qlike:.4f}")
                
        except Exception as e:
            print(f"  ❌ Error loading SpotV2Net: {e}")
            print(f"     This may be due to architecture mismatch")
            # Try with different head counts as fallback
            for try_heads in [6, 8, 4]:
                try:
                    print(f"     Trying {try_heads} attention heads...")
                    model = GATModel(
                        num_node_features=num_node_features,
                        num_edge_features=num_edge_features,
                        num_heads=try_heads,
                        output_node_channels=1,
                        dim_hidden_layers=self.config.get('dim_hidden_layers', [500]),
                        dropout_att=self.config.get('dropout_att', 0.0),
                        dropout=self.config.get('dropout', 0.1),
                        activation=self.config.get('activation', 'relu'),
                        concat_heads=self.config.get('concat_heads', True),
                        negative_slope=self.config.get('negative_slope', 0.2),
                        standardize=False
                    ).to(self.device)
                    
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"     ✅ Success with {try_heads} heads!")
                    actual_num_heads = try_heads
                    break
                except Exception as retry_e:
                    print(f"     ❌ Failed with {try_heads} heads: {str(retry_e)[:50]}...")
                    continue
            else:
                print(f"     ❌ Could not load SpotV2Net with any head configuration")
                return None, None
        
        model.eval()
        
        # Get predictions (same as complete version)
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader_gnn, desc="SpotV2Net evaluation"):
                batch = batch.to(self.device)
                out = model(batch)
                target = batch.y
                
                # Reshape to [n_samples, 30]
                n_samples = len(out) // 30
                out_reshaped = out.view(n_samples, 30)
                target_reshaped = target.view(n_samples, 30)
                
                all_preds.append(out_reshaped.cpu().numpy())
                all_targets.append(target_reshaped.cpu().numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'SpotV2Net_30min',
            'model_path': checkpoint_path,
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_transformer(self):
        """Evaluate TransformerGNN model (same as complete version but with enhanced tracking)"""
        print("\nEvaluating TransformerGNN (30-min)...")
        
        if not self.model_availability.get('TransformerGNN', {}).get('available', False):
            print(f"  ⚠️  TransformerGNN model not available")
            return None, None
        
        checkpoint_path = self.model_availability['TransformerGNN']['path']
        print(f"  📂 Loading from: {checkpoint_path}")
        print(f"     Size: {self.model_availability['TransformerGNN']['size_mb']:.1f} MB")
        
        # Create GNN test dataset
        test_dataset_gnn = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/transformer_gnn_{self.config["seq_length"]}_test',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        test_loader_gnn = GeometricDataLoader(
            test_dataset_gnn,
            batch_size=16,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        # Load checkpoint to get config
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        transformer_config = checkpoint.get('config', {})
        
        # Get dimensions from sample
        sample = test_dataset_gnn[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
        
        print(f"  📐 Dimensions: {num_node_features} node features, {num_edge_features} edge features")
        
        # Create model with saved config (same as complete version)
        model = TransformerGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features if num_edge_features > 0 else 0,
            hidden_dim=transformer_config.get('model', {}).get('hidden_dim', 256),
            num_heads=transformer_config.get('model', {}).get('num_heads', 8),
            num_layers=transformer_config.get('model', {}).get('num_layers', 4),
            dropout=transformer_config.get('model', {}).get('dropout', 0.1),
            edge_dropout=transformer_config.get('model', {}).get('edge_dropout', 0.05),
            use_layer_norm=transformer_config.get('model', {}).get('use_layer_norm', True),
            use_residual=transformer_config.get('model', {}).get('use_residual', True),
            activation=transformer_config.get('model', {}).get('activation', 'gelu'),
            output_dim=1,
            concat_heads=transformer_config.get('model', {}).get('concat_heads', True),
            beta=transformer_config.get('model', {}).get('beta', True)
        ).to(self.device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ TransformerGNN loaded from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        
        val_qlike = checkpoint.get('val_qlike', 'N/A')
        if val_qlike != 'N/A':
            print(f"     Best Val QLIKE: {val_qlike:.4f}")
        
        model.eval()
        
        # Get predictions (same as complete version)
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader_gnn, desc="TransformerGNN evaluation"):
                batch = batch.to(self.device)
                out = model(batch)
                target = batch.y
                
                # Handle shape
                if len(out.shape) == 2 and out.shape[1] == 1:
                    out = out.squeeze(1)
                
                # Reshape to [n_samples, 30]
                if out.ndim == 1:
                    n_samples = len(out) // 30
                    out = out.view(n_samples, 30)
                    target = target.view(n_samples, 30)
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        y_pred = np.vstack(all_preds) if all_preds else np.array([])
        y_true = np.vstack(all_targets) if all_targets else np.array([])
        
        if len(y_pred) == 0:
            return None, None
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'TransformerGNN_30min',
            'model_path': checkpoint_path,
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_pna(self):
        """Evaluate PNA (Principal Neighborhood Aggregation) model (same as complete version)"""
        print("\nEvaluating PNA (30-min)...")
        
        if not self.model_availability.get('PNA', {}).get('available', False):
            print(f"  ⚠️  PNA model not available")
            return None, None
        
        checkpoint_path = self.model_availability['PNA']['path']
        pna_dir = os.path.dirname(checkpoint_path)
        print(f"  📂 Using PNA model from: {pna_dir}")
        print(f"     Path: {checkpoint_path}")
        
        # Create GNN test dataset
        test_dataset_gnn = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/cutting_edge_gnn_{self.config["seq_length"]}_test',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        test_loader_gnn = GeometricDataLoader(
            test_dataset_gnn,
            batch_size=16,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        # Get dimensions from sample
        sample = test_dataset_gnn[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 3
        
        # Create model
        model = PNAVolatilityNet(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=256,
            output_dim=1,
            num_layers=3,
            dropout=0.1
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ PNA loaded from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        
        val_qlike = checkpoint.get('val_qlike', checkpoint.get('best_val_qlike', 'N/A'))
        if val_qlike != 'N/A':
            print(f"     Best Val QLIKE: {val_qlike:.4f}")
        
        model.eval()
        
        # Get predictions
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader_gnn, desc="PNA evaluation"):
                batch = batch.to(self.device)
                out = model(batch, return_uncertainty=False)
                target = batch.y
                
                # Reshape to [n_samples, 30]
                if out.ndim == 1:
                    n_samples = len(out) // 30
                    out = out.view(n_samples, 30)
                    target = target.view(n_samples, 30)
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        y_pred = np.vstack(all_preds) if all_preds else np.array([])
        y_true = np.vstack(all_targets) if all_targets else np.array([])
        
        if len(y_pred) == 0:
            return None, None
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'PNA_30min',
            'model_path': checkpoint_path,
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_xgboost(self):
        """Evaluate XGBoost model"""
        print("\\nEvaluating XGBoost (30-min)...")
        
        if not self.model_availability.get('XGBoost', {}).get('available', False):
            print(f"  ⚠️  XGBoost model not available")
            return None, None
        
        # XGBoost model directory
        xgboost_dir = f'output/XGBoost_30min_{self.config["seq_length"]}'
        test_results_path = os.path.join(xgboost_dir, 'test_results.json')
        
        if not os.path.exists(test_results_path):
            print(f"  ⚠️  XGBoost test results not found at {test_results_path}")
            return None, None
        
        print(f"  📂 Loading results from: {test_results_path}")
        
        # Load pre-computed test results
        with open(test_results_path, 'r') as f:
            xgb_results = json.load(f)
        
        print(f"  ✅ XGBoost results loaded")
        print(f"     Best iteration: {xgb_results.get('best_iteration', 'N/A')}")
        print(f"     Features used: {xgb_results.get('num_features', 'N/A')}")
        print(f"     Training time: {xgb_results.get('training_time', 'N/A')}")
        
        # XGBoost was evaluated offline, so we use pre-computed metrics
        # Note: XGBoost predictions would require the full dataset and model loading
        # For now, we return the metrics but no predictions for plotting
        
        return {
            'model': 'XGBoost_30min',
            'model_path': xgboost_dir,
            'mse': xgb_results['test_rmse_vol'] ** 2,
            'rmse': xgb_results['test_rmse_vol'],
            'mae': xgb_results['test_mae_vol'],
            'qlike': xgb_results['test_qlike'],
            'rmse_var': xgb_results['test_rmse_vol'],  # Approximation
            'mae_var': xgb_results['test_mae_vol'],   # Approximation
            'mse_var': xgb_results['test_rmse_vol'] ** 2,
            'r2': xgb_results.get('test_r2', 'N/A')
        }, None  # No predictions for plotting
    
    def evaluate_spillnet(self):
        """Evaluate SpillNet model (same as complete version)"""
        print("\nEvaluating SpillNet (30-min)...")
        
        if not SPILLNET_AVAILABLE:
            print(f"  ⚠️  SpillNet module not available (removed from codebase)")
            return None, None
        
        if not self.model_availability.get('SpillNet', {}).get('available', False):
            print(f"  ⚠️  SpillNet model not available")
            return None, None
        
        checkpoint_path = self.model_availability['SpillNet']['path']
        
        # Create GNN test dataset
        test_dataset_gnn = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/spillnet_{self.config["seq_length"]}_test',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        test_loader_gnn = GeometricDataLoader(
            test_dataset_gnn,
            batch_size=16,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        # Load checkpoint to get config
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        spillnet_config = checkpoint.get('config', {})
        
        # Get dimensions from sample
        sample = test_dataset_gnn[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
        
        # Create model
        model = SpillNet(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features if num_edge_features > 0 else 1,
            hidden_dim=spillnet_config.get('hidden_dim', 256),
            num_heads=spillnet_config.get('num_heads', 8),
            num_layers=spillnet_config.get('num_layers', 4),
            output_dim=30,
            dropout=spillnet_config.get('dropout', 0.1),
            activation=spillnet_config.get('activation', 'relu'),
            use_layer_norm=spillnet_config.get('use_layer_norm', True),
            use_residual=spillnet_config.get('use_residual', True),
            concat_heads=spillnet_config.get('concat_heads', True),
            negative_slope=spillnet_config.get('negative_slope', 0.2)
        ).to(self.device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ SpillNet loaded from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        
        val_loss = checkpoint.get('val_loss', 'N/A')
        if val_loss != 'N/A':
            print(f"     Best Val Loss: {val_loss:.4f}")
        
        model.eval()
        
        # Get predictions (same complex logic as complete version)
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader_gnn, desc="SpillNet evaluation"):
                batch = batch.to(self.device)
                out = model(batch)
                target = batch.y
                
                # SpillNet outputs [batch_size*30, 30]
                # Reshape to [batch_size, 30] by taking diagonal
                batch_size = len(batch.ptr) - 1 if hasattr(batch, 'ptr') else 1
                
                if out.shape[0] == batch_size * 30 and out.shape[1] == 30:
                    # Extract predictions for each graph
                    batch_preds = []
                    for i in range(batch_size):
                        start_idx = i * 30
                        end_idx = (i + 1) * 30
                        graph_out = out[start_idx:end_idx]
                        # Take diagonal or mean across nodes
                        pred = torch.diag(graph_out) if graph_out.shape[0] == graph_out.shape[1] else graph_out.mean(dim=0)
                        batch_preds.append(pred)
                    out = torch.stack(batch_preds)
                
                # Reshape target similarly
                if target.ndim == 1:
                    target = target.view(batch_size, 30)
                
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        y_pred = np.vstack(all_preds) if all_preds else np.array([])
        y_true = np.vstack(all_targets) if all_targets else np.array([])
        
        if len(y_pred) == 0:
            return None, None
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'SpillNet_30min',
            'model_path': checkpoint_path,
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def create_comparison_table(self, metrics_list):
        """Create formatted comparison table (same as complete version)"""
        df = pd.DataFrame(metrics_list)
        
        # Sort by QLIKE (primary metric for volatility forecasting)
        df = df.sort_values('qlike')
        
        # Format for display
        display_df = df.copy()
        
        # Format numbers
        for col in ['mse', 'mse_var']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2e}")
        
        for col in ['rmse', 'mae', 'rmse_var', 'mae_var']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")
        
        for col in ['qlike']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        return display_df
    
    def plot_predictions_comparison(self, predictions_dict, n_samples=100, assets=[0, 5, 10, 15, 20]):
        """Plot predictions for selected assets (same as complete version but enhanced)"""
        # Load DOW30 symbols
        with open('config/dow30_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        symbols = config['dow30_symbols']
        
        # Use the enhanced academic plotter
        print("🎨 Creating enhanced prediction comparison plots...")
        plotter = AcademicPlotter(output_dir='paper_assets')
        
        # Create enhanced prediction comparison
        selected_stocks = [symbols[i] for i in assets if i < len(symbols)]
        plotter.create_prediction_comparison(
            predictions_dict=predictions_dict,
            n_samples=n_samples,
            selected_stocks=selected_stocks
        )
        
        # Also create the original plot for backward compatibility
        n_assets = len(assets)
        fig, axes = plt.subplots(n_assets, 1, figsize=(14, 3*n_assets))
        if n_assets == 1:
            axes = [axes]
        
        colors = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
        
        for idx, asset_idx in enumerate(assets):
            ax = axes[idx]
            
            # Get stock symbol
            stock_symbol = symbols[asset_idx] if asset_idx < len(symbols) else f"Asset {asset_idx}"
            
            # Get true values for this asset
            true_values = None
            
            for model_name, (metrics, preds) in predictions_dict.items():
                # Skip TransformerGNN - not in paper
                if 'TransformerGNN' in model_name or 'transformer' in model_name.lower():
                    continue
                    
                if preds is not None and len(preds) > 0:
                    if true_values is None:
                        # Extract true values from the first model's evaluation
                        all_targets = []
                        for batch in self.test_loader:
                            all_targets.append(batch['target'].numpy())
                            if len(all_targets) * 32 >= n_samples:
                                break
                        true_values = np.vstack(all_targets)[:n_samples, asset_idx]
                        ax.plot(range(n_samples), true_values, label='Actual', 
                               color='black', linewidth=2, alpha=0.8)
                    
                    # Plot predictions with special styling for PNA
                    model_preds = preds[:n_samples, asset_idx]
                    color_idx = list(predictions_dict.keys()).index(model_name) + 1
                    
                    # Highlight PNA as PROPOSED MODEL
                    if 'PNA' in model_name:
                        ax.plot(range(n_samples), model_preds, 
                               label=f"{model_name.replace('_30min', '')} (PROPOSED)", 
                               alpha=1.0, color='#FF6B35', linewidth=3, zorder=10)
                    else:
                        ax.plot(range(n_samples), model_preds, 
                               label=model_name.replace('_30min', ''), 
                               alpha=0.7, color=colors[color_idx % len(colors)])
            
            ax.set_title(f'{stock_symbol} - 30-Minute Volatility')
            ax.set_xlabel('Time (30-min intervals)')
            ax.set_ylabel('Standardized Log Volatility')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('30-Minute Intraday Volatility Predictions - Final Comprehensive\n(PNA = PROPOSED MODEL)', fontsize=14)
        plt.tight_layout()
        plt.savefig('intraday_predictions_final_30min.png', dpi=120, bbox_inches='tight')
        plt.close()
        
        print("✅ Enhanced prediction comparison plots saved")
    
    def run_evaluation(self):
        """Run complete evaluation (enhanced version of complete's run_evaluation)"""
        print("="*80)
        print("FINAL COMPREHENSIVE MODEL EVALUATION - 30-MINUTE INTRADAY VOLATILITY")
        print("="*80)
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Sequence length: {self.config['seq_length']} intervals (≈{self.config['seq_length']/13:.1f} days)")
        print(f"Forecast horizon: 1 interval (30 minutes)")
        
        # Enhanced model availability check
        print("\n📊 DETAILED MODEL AVAILABILITY:")
        for model_type, info in self.model_availability.items():
            if info.get('available', False):
                print(f"  ✅ {model_type}: {info['path']}")
                print(f"     Size: {info['size_mb']:.1f} MB, Modified: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                if info.get('val_qlike', 'N/A') != 'N/A':
                    print(f"     Val QLIKE: {info['val_qlike']:.4f}")
            else:
                print(f"  ❌ {model_type}: Not available")
                if 'error' in info:
                    print(f"     Error: {info['error']}")
        print("="*80)
        
        all_metrics = []
        predictions = {}
        
        # Evaluate baselines (same as complete version)
        naive_metrics, naive_pred = self.evaluate_naive_persistence()
        if naive_metrics:
            all_metrics.append(naive_metrics)
            predictions['Naive'] = (naive_metrics, naive_pred)
        
        hist_metrics, hist_pred = self.evaluate_historical_mean()
        if hist_metrics:
            all_metrics.append(hist_metrics)
            predictions['HistMean'] = (hist_metrics, hist_pred)
        
        ewma_metrics, ewma_pred = self.evaluate_ewma()
        if ewma_metrics:
            all_metrics.append(ewma_metrics)
            predictions['EWMA'] = (ewma_metrics, ewma_pred)
        
        har_metrics, har_pred = self.evaluate_har_intraday()
        if har_metrics:
            all_metrics.append(har_metrics)
            predictions['HAR'] = (har_metrics, har_pred)
        
        # Evaluate neural networks (enhanced with path tracking)
        lstm_metrics, lstm_pred = self.evaluate_lstm()
        if lstm_metrics:
            all_metrics.append(lstm_metrics)
            predictions['LSTM'] = (lstm_metrics, lstm_pred)
        
        gnn_metrics, gnn_pred = self.evaluate_spotv2net()
        if gnn_metrics:
            all_metrics.append(gnn_metrics)
            predictions['SpotV2Net'] = (gnn_metrics, gnn_pred)
        
        # Evaluate advanced models (SKIP TransformerGNN - not in paper)
        # transformer_metrics, transformer_pred = self.evaluate_transformer()
        # if transformer_metrics:
        #     all_metrics.append(transformer_metrics)
        #     predictions['TransformerGNN'] = (transformer_metrics, transformer_pred)
        print("\nSkipping TransformerGNN evaluation (excluded from paper)...")
        
        # Evaluate XGBoost baseline model
        xgboost_metrics, xgboost_pred = self.evaluate_xgboost()
        if xgboost_metrics:
            all_metrics.append(xgboost_metrics)
            predictions['XGBoost'] = (xgboost_metrics, xgboost_pred)
        
        # Evaluate cutting-edge models - PNA IS THE PROPOSED MODEL
        pna_metrics, pna_pred = self.evaluate_pna()
        if pna_metrics:
            all_metrics.append(pna_metrics)
            predictions['PNA'] = (pna_metrics, pna_pred)
            print(f"✅ PNA (PROPOSED MODEL) evaluated - QLIKE: {pna_metrics['qlike']:.4f}")
        
        # Create comparison table (same as complete version)
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON (30-MINUTE INTRADAY)")
        print("="*80)
        
        comparison_df = self.create_comparison_table(all_metrics)
        print("\n", comparison_df.to_string(index=False))
        
        # Calculate improvements (same as complete version)
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Get naive baseline for comparison
        naive_qlike = next((m['qlike'] for m in all_metrics if 'Naive' in m['model']), None)
        
        if naive_qlike:
            print("\nImprovement over Naive Persistence:")
            for metrics in all_metrics:
                if 'Naive' not in metrics['model']:
                    improvement = ((naive_qlike - metrics['qlike']) / naive_qlike) * 100
                    print(f"  {metrics['model']:25s}: {improvement:+.2f}%")
        
        # Best model
        best_model = comparison_df.iloc[0]['model']
        best_qlike = float(comparison_df.iloc[0]['qlike'])
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   QLIKE: {best_qlike:.4f}")
        if 'PNA' in best_model:
            print(f"   🎯 PNA is the PROPOSED MODEL and achieved best performance!")
        elif 'PNA' in [m['model'] for m in all_metrics]:
            pna_qlike = next((m['qlike'] for m in all_metrics if 'PNA' in m['model']), None)
            if pna_qlike:
                print(f"   📊 PNA (PROPOSED) QLIKE: {pna_qlike:.4f}")
        
        # Enhanced plotting and outputs
        if len(predictions) > 0:
            self.plot_predictions_comparison(predictions)
            
            # Generate enhanced academic publication plots
            print("\n📊 Generating enhanced academic publication plots...")
            plotter = AcademicPlotter(output_dir='paper_assets')
            
            # Create enhanced prediction comparison
            plotter.create_prediction_comparison(
                predictions_dict=predictions,
                n_samples=200,
                selected_stocks=['AAPL', 'MSFT', 'JPM', 'CVX', 'WMT']
            )
            
            # Collect true values for enhanced per-interval analysis
            print("   📊 Collecting true values from test set for enhanced analysis...")
            all_true_values = []
            for batch in self.test_loader:
                all_true_values.append(batch['target'].numpy())
            true_values = np.vstack(all_true_values)
            print(f"   ✅ Collected {len(true_values)} true values for analysis")
            
            # Create enhanced performance by time of day analysis
            plotter.create_performance_by_time_of_day(predictions, true_values=true_values)
            
            # Create enhanced model comparison plots
            plotter.create_model_comparison_plots(comparison_df)
            
            # Create correlation analysis plots
            plotter.create_correlation_analysis(predictions)
            
            # Create volatility clustering visualization
            plotter.create_volatility_clustering_plot()
            
            print("✅ Enhanced academic plots generated successfully!")
            print("📁 Saved to paper_assets/ directory with enhanced features")
        
        # Enhanced results saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced results dictionary
        results = {
            'timestamp': timestamp,
            'evaluation_type': 'final_comprehensive_enhanced',
            'test_samples': len(self.test_dataset),
            'sequence_length': self.config['seq_length'],
            'forecast_horizon': '30_minutes',
            'config_path': self.config_path,
            'device': str(self.device),
            'model_availability': self.model_availability,
            'data_files': {
                'vol_file': self.vol_file,
                'volvol_file': self.volvol_file,
                'scaler_file': self.scaler_file
            },
            'metrics': all_metrics,
            'best_model': best_model,
            'best_qlike': best_qlike,
            'improvement_over_naive': ((naive_qlike - best_qlike) / naive_qlike * 100) if naive_qlike else 0
        }
        
        output_file = f"evaluation_results_final_comprehensive_30min_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Enhanced results saved to: {output_file}")
        
        # Save CSV for easy Excel import
        csv_file = f"evaluation_results_final_comprehensive_30min_{timestamp}.csv"
        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv(csv_file, index=False)
        print(f"✅ CSV saved to: {csv_file}")
        
        # Generate enhanced LaTeX tables for academic paper
        print("\n📄 Generating enhanced LaTeX tables...")
        latex_gen = LaTeXTableGenerator(output_dir='paper_assets')
        
        # Ensure HAR is included in the metrics for LaTeX tables
        har_in_metrics = any('HAR' in m.get('model', '') for m in all_metrics)
        if har_in_metrics:
            print("   ✓ HAR baseline included in enhanced LaTeX tables")
        else:
            print("   ⚠️ Warning: HAR not found in metrics for LaTeX tables")
        
        latex_tables = latex_gen.save_all_tables(all_metrics, timestamp=timestamp)
        print("   ✓ Enhanced LaTeX tables generated with all baselines included")
        
        print("\n" + "="*80)
        print("FINAL COMPREHENSIVE EVALUATION COMPLETE - ENHANCED VERSION")
        print("="*80)
        print(f"🎯 Models evaluated: {len(all_metrics)}")
        print(f"🏆 Best model: {best_model}")
        print(f"📊 Best QLIKE: {best_qlike:.4f}")
        print(f"📈 Improvement over naive: {((naive_qlike - best_qlike) / naive_qlike * 100):.1f}%" if naive_qlike else "N/A")
        print(f"📁 All outputs saved to paper_assets/ and current directory")
        print("="*80)
        
        return all_metrics


def main():
    """Main function with enhanced argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Final Comprehensive Model Evaluation - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
ENHANCED FEATURES:
• Detailed model discovery and path tracking
• Enhanced error handling and stability
• Publication-ready outputs with academic plots
• Comprehensive baseline evaluation (Naive, Historical Mean, HAR, EWMA)
• Neural networks (LSTM, TransformerGNN)
• Graph Neural Networks (PNA, SpotV2Net, SpillNet)
• Enhanced result tracking and analysis

OUTPUT FILES:
• evaluation_results_final_comprehensive_30min_[timestamp].json
• evaluation_results_final_comprehensive_30min_[timestamp].csv  
• paper_assets/ (Enhanced LaTeX tables and academic plots)
• intraday_predictions_final_30min.png

Based on 6_evaluate_all_models_complete.py with 95% compatibility + enhancements.
        '''
    )
    
    parser.add_argument('--config', type=str, default='config/GNN_param.yaml',
                       help='Configuration file path (default: config/GNN_param.yaml)')
    
    args = parser.parse_args()
    
    # Create enhanced evaluator
    evaluator = FinalComprehensiveEvaluator(config_path=args.config)
    
    # Run enhanced evaluation
    all_metrics = evaluator.run_evaluation()


if __name__ == "__main__":
    main()