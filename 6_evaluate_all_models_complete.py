#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Including Transformer and SpillNet
==================================================================
Evaluates all models on the same test set with proper metrics
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


class ComprehensiveEvaluator:
    """Unified evaluator for all volatility forecasting models"""
    
    def __init__(self, config_path='config/GNN_param.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Update for 30-minute data
        self.config['seq_length'] = 42
        
        # Data files
        self.vol_file = 'processed_data/vols_mats_30min_standardized.h5'
        self.volvol_file = 'processed_data/volvols_mats_30min_standardized.h5'
        self.scaler_file = 'processed_data/vols_30min_mean_std_scalers.csv'
        
        # Check files exist
        for file_path, name in [(self.vol_file, 'Volatility'), 
                                (self.volvol_file, 'Vol-of-vol'),
                                (self.scaler_file, 'Scaler')]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{name} file not found: {file_path}")
        
        # Initialize evaluator
        self.evaluator = VolatilityEvaluator(self.scaler_file)
        
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load test dataset
        self.load_test_data()
        
    def load_test_data(self):
        """Load test data for all models"""
        print("Loading 30-minute test data...")
        
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
        
        print(f"Test set: {len(self.test_dataset)} samples")
        
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
        
    def evaluate_naive_persistence(self):
        """Naive baseline: predict next = current"""
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
        """Historical mean baseline"""
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
    
    def evaluate_ewma(self):
        """EWMA (Exponentially Weighted Moving Average) baseline"""
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
        """Evaluate LSTM model"""
        print("\nEvaluating LSTM (30-min)...")
        
        # Check if model exists
        lstm_dir = f'output/LSTM_30min_{self.config["seq_length"]}'
        checkpoint_path = os.path.join(lstm_dir, 'best_model.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  LSTM model not found at {checkpoint_path}")
            return None, None
        
        # Load LSTM model
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
        print(f"  Loaded LSTM from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        
        model.eval()
        
        # Get predictions
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
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_spotv2net(self):
        """Evaluate SpotV2Net (GNN) model"""
        print("\nEvaluating SpotV2Net (30-min)...")
        
        # Check if model exists
        gnn_dir = f'output/{self.config["modelname"]}_30min_{self.config["seq_length"]}'
        checkpoint_path = os.path.join(gnn_dir, 'best_model.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  SpotV2Net model not found at {checkpoint_path}")
            return None, None
        
        # Create GNN test dataset
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
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        # Get dimensions from sample
        sample = test_dataset_gnn[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 1
        
        # Create model
        model = GATModel(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_heads=self.config['num_heads'],
            output_node_channels=1,
            dim_hidden_layers=self.config['dim_hidden_layers'],
            dropout_att=self.config['dropout_att'],
            dropout=self.config['dropout'],
            activation=self.config['activation'],
            concat_heads=self.config['concat_heads'],
            negative_slope=self.config['negative_slope'],
            standardize=False
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded SpotV2Net from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        
        model.eval()
        
        # Get predictions
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
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_transformer(self):
        """Evaluate TransformerGNN model"""
        print("\nEvaluating TransformerGNN (30-min)...")
        
        # Check if model exists
        transformer_dir = f'output/TransformerGNN_30min_{self.config["seq_length"]}'
        checkpoint_path = os.path.join(transformer_dir, 'best_model.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  TransformerGNN model not found at {checkpoint_path}")
            return None, None
        
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
        
        # Create model with saved config
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
        print(f"  Loaded TransformerGNN from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        print(f"  Best Val QLIKE: {checkpoint.get('val_qlike', 'N/A'):.4f}")
        
        model.eval()
        
        # Get predictions
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
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_pna(self):
        """Evaluate PNA (Principal Neighborhood Aggregation) model"""
        print("\nEvaluating PNA (30-min)...")
        
        # Check if model exists - look for the most recent PNA model
        import glob
        pna_models = glob.glob('output/pna_30min_*/best_model.pt')
        
        if not pna_models:
            print(f"  ⚠️  PNA model not found")
            return None, None
        
        # Use the most recent model
        checkpoint_path = sorted(pna_models)[-1]
        pna_dir = os.path.dirname(checkpoint_path)
        print(f"  Using PNA model from: {pna_dir}")
        
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
        print(f"  Loaded PNA from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        print(f"  Best Val QLIKE: {checkpoint.get('val_qlike', checkpoint.get('best_val_qlike', 'N/A'))}")
        
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
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def evaluate_spillnet(self):
        """Evaluate SpillNet model"""
        print("\nEvaluating SpillNet (30-min)...")
        
        if not SPILLNET_AVAILABLE:
            print(f"  ⚠️  SpillNet module not available (removed from codebase)")
            return None, None
        
        # Check if model exists
        spillnet_dir = f'output/SpillNet_30min_{self.config["seq_length"]}'
        checkpoint_path = os.path.join(spillnet_dir, 'best_model.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  SpillNet model not found at {checkpoint_path}")
            return None, None
        
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
        print(f"  Loaded SpillNet from epoch {checkpoint.get('epoch', 'N/A') + 1}")
        print(f"  Best Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        model.eval()
        
        # Get predictions
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
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }, y_pred
    
    def create_comparison_table(self, metrics_list):
        """Create formatted comparison table"""
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
        """Plot predictions for selected assets"""
        # Load DOW30 symbols
        with open('config/dow30_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        symbols = config['dow30_symbols']
        
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
                    
                    # Plot predictions
                    model_preds = preds[:n_samples, asset_idx]
                    color_idx = list(predictions_dict.keys()).index(model_name) + 1
                    ax.plot(range(n_samples), model_preds, 
                           label=model_name.replace('_30min', ''), 
                           alpha=0.7, color=colors[color_idx % len(colors)])
            
            ax.set_title(f'{stock_symbol} - 30-Minute Volatility')
            ax.set_xlabel('Time (30-min intervals)')
            ax.set_ylabel('Standardized Log Volatility')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('30-Minute Intraday Volatility Predictions - All Models', fontsize=14)
        plt.tight_layout()
        plt.savefig('intraday_predictions_all_models_30min.png', dpi=120, bbox_inches='tight')
        plt.close()
        
        print("✅ Saved prediction comparison plots")
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("="*80)
        print("COMPREHENSIVE MODEL EVALUATION - 30-MINUTE INTRADAY VOLATILITY")
        print("="*80)
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Sequence length: {self.config['seq_length']} intervals (≈{self.config['seq_length']/13:.1f} days)")
        print(f"Forecast horizon: 1 interval (30 minutes)")
        
        # Check which models are available
        print("\nChecking available models:")
        import glob
        lstm_available = os.path.exists(f'output/LSTM_30min_{self.config["seq_length"]}/best_model.pt')
        gnn_available = os.path.exists(f'output/{self.config["modelname"]}_30min_{self.config["seq_length"]}/best_model.pt')
        transformer_available = os.path.exists(f'output/TransformerGNN_30min_{self.config["seq_length"]}/best_model.pt')
        spillnet_available = os.path.exists(f'output/SpillNet_30min_{self.config["seq_length"]}/best_model.pt')
        pna_available = len(glob.glob('output/pna_30min_*/best_model.pt')) > 0
        
        print(f"  LSTM: {'✓ Available' if lstm_available else '✗ Not found'}")
        print(f"  SpotV2Net (GAT): {'✓ Available' if gnn_available else '✗ Not found'}")
        print(f"  TransformerGNN: {'✓ Available' if transformer_available else '✗ Not found'}")
        print(f"  SpillNet: {'✓ Available' if spillnet_available else '✗ Not found'}")
        print(f"  PNA: {'✓ Available' if pna_available else '✗ Not found'}")
        print("="*80)
        
        all_metrics = []
        predictions = {}
        
        # Evaluate baselines
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
        
        # Evaluate neural networks
        lstm_metrics, lstm_pred = self.evaluate_lstm()
        if lstm_metrics:
            all_metrics.append(lstm_metrics)
            predictions['LSTM'] = (lstm_metrics, lstm_pred)
        
        gnn_metrics, gnn_pred = self.evaluate_spotv2net()
        if gnn_metrics:
            all_metrics.append(gnn_metrics)
            predictions['SpotV2Net'] = (gnn_metrics, gnn_pred)
        
        # Evaluate advanced models
        transformer_metrics, transformer_pred = self.evaluate_transformer()
        if transformer_metrics:
            all_metrics.append(transformer_metrics)
            predictions['TransformerGNN'] = (transformer_metrics, transformer_pred)
        
        spillnet_metrics, spillnet_pred = self.evaluate_spillnet()
        if spillnet_metrics:
            all_metrics.append(spillnet_metrics)
            predictions['SpillNet'] = (spillnet_metrics, spillnet_pred)
        
        # Evaluate cutting-edge models
        pna_metrics, pna_pred = self.evaluate_pna()
        if pna_metrics:
            all_metrics.append(pna_metrics)
            predictions['PNA'] = (pna_metrics, pna_pred)
        
        # Create comparison table
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON (30-MINUTE INTRADAY)")
        print("="*80)
        
        comparison_df = self.create_comparison_table(all_metrics)
        print("\n", comparison_df.to_string(index=False))
        
        # Calculate improvements
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
        
        # Plot predictions
        if len(predictions) > 0:
            self.plot_predictions_comparison(predictions)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'test_samples': len(self.test_dataset),
            'sequence_length': self.config['seq_length'],
            'forecast_horizon': '30_minutes',
            'metrics': all_metrics,
            'best_model': best_model,
            'best_qlike': best_qlike
        }
        
        output_file = f"evaluation_results_all_models_30min_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Save CSV for easy Excel import
        csv_file = f"evaluation_results_all_models_30min_{timestamp}.csv"
        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv(csv_file, index=False)
        print(f"✅ CSV saved to: {csv_file}")
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE - ALL MODELS INCLUDED")
        print("="*80)
        
        return all_metrics


def main():
    evaluator = ComprehensiveEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()