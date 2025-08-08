#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Suite for SpotV2Net Research
============================================================

This script provides a unified evaluation framework for comparing:
1. LSTM baseline
2. SpotV2Net (GAT-based GNN)
3. Naive persistence baseline
4. Statistical baselines (GARCH, etc. if needed)

All models are evaluated on the same test set with consistent metrics.
Results are presented in both standardized and original scales.
"""

import torch
import numpy as np
import pandas as pd
import h5py
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
import warnings
warnings.filterwarnings('ignore')

# Import models
import sys
sys.path.append('.')
from utils.models import GATModel
from utils.dataset import CovarianceLaggedDataset


class ModelEvaluator:
    """Unified evaluation framework for all models"""
    
    def __init__(self, seq_length: int = 42):
        self.seq_length = seq_length
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load scaler parameters
        self.load_scaler_params()
        
        # Results storage
        self.results = {}
        
    def load_scaler_params(self):
        """Load standardization parameters for inverse transform"""
        scaler_file = 'processed_data/vols_mean_std_scalers.csv'
        if os.path.exists(scaler_file):
            params = pd.read_csv(scaler_file)
            self.vol_log_mean = params['Vol_LogMean'].values[0]
            self.vol_log_std = params['Vol_LogStd'].values[0]
            self.cov_mean = params['Cov_Mean'].values[0]
            self.cov_std = params['Cov_Std'].values[0]
            print(f"Loaded scaler params for inverse transform")
        else:
            print("Warning: Scaler params not found, using defaults")
            self.vol_log_mean = 0.0
            self.vol_log_std = 1.0
            self.cov_mean = 0.0
            self.cov_std = 1.0
    
    def inverse_transform_volatilities(self, standardized_vols):
        """
        Inverse transform volatilities from standardized to original scale
        
        Args:
            standardized_vols: Array of standardized volatility values
            
        Returns:
            Original scale volatilities
        """
        # Inverse standardize then exp
        vols_log = standardized_vols * self.vol_log_std + self.vol_log_mean
        vols_original = np.exp(vols_log)
        return vols_original
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                       model_name: str) -> Dict:
        """
        Compute comprehensive metrics for model evaluation
        
        Args:
            predictions: Model predictions (standardized scale)
            targets: True values (standardized scale)
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary of metrics
        """
        # Ensure numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Flatten if needed
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        
        # Remove samples with NaN values
        valid_mask = ~(np.any(np.isnan(predictions), axis=1) | np.any(np.isnan(targets), axis=1))
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        if len(predictions) == 0:
            print(f"Warning: No valid predictions for {model_name} after removing NaN")
            return {
                'model': model_name,
                'mse_standardized': float('nan'),
                'rmse_standardized': float('nan'),
                'mae_standardized': float('nan'),
                'r2_standardized': float('nan'),
                'vol_rmse_standardized': float('nan'),
                'vol_mae_standardized': float('nan'),
                'vol_rmse_original': float('nan'),
                'vol_mae_original': float('nan'),
                'vol_mape': float('nan'),
                'direction_accuracy': float('nan'),
                'n_samples': 0
            }
        
        # Extract volatilities (first 30 values of each 900-dim vector)
        vol_pred_std = predictions[:, :30]
        vol_true_std = targets[:, :30]
        
        # Standardized scale metrics
        mse_std = np.mean((predictions - targets) ** 2)
        rmse_std = np.sqrt(mse_std)
        mae_std = np.mean(np.abs(predictions - targets))
        
        # Volatility-specific standardized metrics
        vol_mse_std = np.mean((vol_pred_std - vol_true_std) ** 2)
        vol_rmse_std = np.sqrt(vol_mse_std)
        vol_mae_std = np.mean(np.abs(vol_pred_std - vol_true_std))
        
        # R² score (standardized)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_std = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Transform to original scale (volatilities only)
        vol_pred_orig = self.inverse_transform_volatilities(vol_pred_std)
        vol_true_orig = self.inverse_transform_volatilities(vol_true_std)
        
        # Original scale metrics (volatilities)
        vol_mse_orig = np.mean((vol_pred_orig - vol_true_orig) ** 2)
        vol_rmse_orig = np.sqrt(vol_mse_orig)
        vol_mae_orig = np.mean(np.abs(vol_pred_orig - vol_true_orig))
        
        # Percentage error in original scale
        vol_mape = np.mean(np.abs((vol_true_orig - vol_pred_orig) / (vol_true_orig + 1e-8))) * 100
        
        # Directional accuracy (did we predict the direction of change correctly?)
        if predictions.shape[0] > 1:
            pred_changes = np.diff(vol_pred_std, axis=0)
            true_changes = np.diff(vol_true_std, axis=0)
            direction_accuracy = np.mean(np.sign(pred_changes) == np.sign(true_changes))
        else:
            direction_accuracy = 0.0
        
        metrics = {
            'model': model_name,
            'mse_standardized': float(mse_std),
            'rmse_standardized': float(rmse_std),
            'mae_standardized': float(mae_std),
            'r2_standardized': float(r2_std),
            'vol_rmse_standardized': float(vol_rmse_std),
            'vol_mae_standardized': float(vol_mae_std),
            'vol_rmse_original': float(vol_rmse_orig),
            'vol_mae_original': float(vol_mae_orig),
            'vol_mape': float(vol_mape),
            'direction_accuracy': float(direction_accuracy),
            'n_samples': len(predictions)
        }
        
        return metrics
    
    def evaluate_lstm(self, model_dir: str = 'output/LSTM_42_fixed') -> Optional[Dict]:
        """Evaluate LSTM model"""
        print("\n" + "="*60)
        print("Evaluating LSTM Model")
        print("="*60)
        
        # Try multiple possible locations
        possible_dirs = [model_dir, 'output/LSTM_42', 'output/LSTM_42_fixed']
        model_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(os.path.join(dir_path, 'best_model.pt')):
                model_dir = dir_path
                break
        
        if model_dir is None:
            print(f"LSTM model not found in any of: {possible_dirs}")
            return None
        
        print(f"Loading LSTM from: {model_dir}")
        
        # Import LSTM model from the fixed version
        from torch import nn
        
        class LSTMModel(nn.Module):
            """LSTM for volatility prediction"""
            def __init__(self, input_size=900, hidden_size=256, num_layers=2, dropout=0.2):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, 900)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out
        
        # Load test data
        vols_file = 'processed_data/vols_mats_taq_standardized.h5'
        sequences = []
        targets = []
        
        with h5py.File(vols_file, 'r') as f:
            sorted_keys = sorted(f.keys(), key=int)
            matrices = [f[key][:].flatten() for key in sorted_keys]
        
        for i in range(len(matrices) - self.seq_length):
            seq = np.array(matrices[i:i+self.seq_length])
            target = matrices[i+self.seq_length]
            sequences.append(seq)
            targets.append(target)
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        # Get test split (same as training)
        seq_offset = self.seq_length - 1
        train_end_idx = max(0, min(1008 - seq_offset, len(X)))
        val_end_idx = max(0, min(1260 - seq_offset, len(X)))
        
        X_test = X[val_end_idx:]
        y_test = y[val_end_idx:]
        
        if len(X_test) == 0:
            print("No test data available for LSTM")
            return None
        
        # Create test loader
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Load model
        model = LSTMModel(input_size=900, hidden_size=256, num_layers=2, dropout=0.2).to(self.device)
        checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'), 
                              map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get predictions
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader, desc="LSTM evaluation"):
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets, 'LSTM')
        self.results['LSTM'] = metrics
        
        print(f"LSTM Test Samples (after NaN removal): {metrics['n_samples']}")
        if metrics['n_samples'] > 0:
            print(f"LSTM RMSE (standardized): {metrics['rmse_standardized']:.6f}")
            print(f"LSTM Vol RMSE (original): {metrics['vol_rmse_original']:.6f}")
        
        return metrics
    
    def evaluate_gnn(self, model_dir: str = 'output/20240525_RGNN_std_optuna_42') -> Optional[Dict]:
        """Evaluate SpotV2Net GNN model"""
        print("\n" + "="*60)
        print("Evaluating SpotV2Net (GNN) Model")
        print("="*60)
        
        if not os.path.exists(os.path.join(model_dir, 'best_model.pt')):
            print(f"GNN model not found in {model_dir}")
            return None
        
        # Load config
        import yaml
        with open('config/GNN_param.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create dataset
        dataset = CovarianceLaggedDataset(
            hdf5_file1='processed_data/vols_mats_taq_standardized.h5',
            hdf5_file2='processed_data/volvols_mats_taq_standardized.h5',
            root=f'processed_data/vols_mats_taq_standardized_test_{self.seq_length}',
            seq_length=self.seq_length
        )
        
        # Get test split
        seq_offset = self.seq_length - 1
        train_end_idx = max(0, min(1008 - seq_offset, len(dataset)))
        val_end_idx = max(0, min(1260 - seq_offset, len(dataset)))
        
        test_dataset = dataset[val_end_idx:]
        
        if len(test_dataset) == 0:
            print("No test data available for GNN")
            return None
        
        test_loader = GeometricDataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Get sample to determine dimensions
        sample = test_dataset[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else 1
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 1
        
        # Load model
        model = GATModel(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_heads=config['num_heads'],
            output_node_channels=config['output_node_channels'],
            dim_hidden_layers=config['dim_hidden_layers'],
            dropout_att=config['dropout_att'],
            dropout=config['dropout'],
            activation=config['activation'],
            concat_heads=config['concat_heads'],
            negative_slope=config['negative_slope'],
            standardize=False
        ).to(self.device)
        
        checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'), 
                              map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get predictions
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="GNN evaluation"):
                batch = batch.to(self.device)
                outputs = model(batch)
                target = batch.y_x if hasattr(batch, 'y_x') else batch.y
                
                # GNN outputs are node-level (30 values), need to pad to 900
                batch_size = batch.num_graphs
                outputs_padded = torch.zeros(batch_size, 900)
                outputs_padded[:, :30] = outputs.reshape(batch_size, -1)
                
                target_padded = torch.zeros(batch_size, 900)
                target_padded[:, :30] = target.reshape(batch_size, -1)
                
                all_predictions.append(outputs_padded.cpu().numpy())
                all_targets.append(target_padded.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets, 'SpotV2Net')
        self.results['SpotV2Net'] = metrics
        
        print(f"SpotV2Net Test Samples (after NaN removal): {metrics['n_samples']}")
        if metrics['n_samples'] > 0:
            print(f"SpotV2Net RMSE (standardized): {metrics['rmse_standardized']:.6f}")
            print(f"SpotV2Net Vol RMSE (original): {metrics['vol_rmse_original']:.6f}")
        
        return metrics
    
    def evaluate_naive_baseline(self) -> Dict:
        """Evaluate naive persistence baseline (tomorrow = today)"""
        print("\n" + "="*60)
        print("Evaluating Naive Persistence Baseline")
        print("="*60)
        
        # Load data
        vols_file = 'processed_data/vols_mats_taq_standardized.h5'
        
        with h5py.File(vols_file, 'r') as f:
            sorted_keys = sorted(f.keys(), key=int)
            matrices = [f[key][:].flatten() for key in sorted_keys]
        
        # Get test split indices
        seq_offset = self.seq_length - 1
        train_end_idx = max(0, min(1008 - seq_offset, len(matrices) - self.seq_length))
        val_end_idx = max(0, min(1260 - seq_offset, len(matrices) - self.seq_length))
        
        # For naive baseline, prediction is just the previous value
        test_start = val_end_idx + self.seq_length
        test_end = len(matrices)
        
        if test_end <= test_start + 1:
            print("Not enough test data for naive baseline")
            return None
        
        predictions = []
        targets = []
        
        for i in range(test_start, test_end - 1):
            # Predict that tomorrow = today
            predictions.append(matrices[i])
            targets.append(matrices[i + 1])
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets, 'Naive')
        self.results['Naive'] = metrics
        
        print(f"Naive Test Samples (after NaN removal): {metrics['n_samples']}")
        if metrics['n_samples'] > 0:
            print(f"Naive RMSE (standardized): {metrics['rmse_standardized']:.6f}")
            print(f"Naive Vol RMSE (original): {metrics['vol_rmse_original']:.6f}")
        
        return metrics
    
    def evaluate_mean_baseline(self) -> Dict:
        """Evaluate historical mean baseline"""
        print("\n" + "="*60)
        print("Evaluating Historical Mean Baseline")
        print("="*60)
        
        # Load data
        vols_file = 'processed_data/vols_mats_taq_standardized.h5'
        
        with h5py.File(vols_file, 'r') as f:
            sorted_keys = sorted(f.keys(), key=int)
            matrices = [f[key][:].flatten() for key in sorted_keys]
        
        matrices = np.array(matrices)
        
        # Get splits
        seq_offset = self.seq_length - 1
        train_end_idx = max(0, min(1008 - seq_offset, len(matrices) - self.seq_length))
        val_end_idx = max(0, min(1260 - seq_offset, len(matrices) - self.seq_length))
        
        # Compute training mean
        train_mean = np.mean(matrices[:train_end_idx + seq_offset], axis=0)
        
        # Test predictions are all the training mean
        test_start = val_end_idx + self.seq_length
        test_end = len(matrices)
        
        predictions = np.tile(train_mean, (test_end - test_start, 1))
        targets = matrices[test_start:test_end]
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets, 'Mean')
        self.results['Mean'] = metrics
        
        print(f"Mean Test Samples (after NaN removal): {metrics['n_samples']}")
        if metrics['n_samples'] > 0:
            print(f"Mean RMSE (standardized): {metrics['rmse_standardized']:.6f}")
            print(f"Mean Vol RMSE (original): {metrics['vol_rmse_original']:.6f}")
        
        return metrics
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of all models"""
        if not self.results:
            print("No results to compare")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(self.results).T
        
        # Select key metrics for comparison
        key_metrics = [
            'rmse_standardized',
            'vol_rmse_standardized', 
            'vol_rmse_original',
            'vol_mae_original',
            'vol_mape',
            'r2_standardized',
            'direction_accuracy'
        ]
        
        comparison_df = df[key_metrics].round(4)
        
        # Add relative performance vs naive
        if 'Naive' in self.results:
            naive_rmse = self.results['Naive']['vol_rmse_original']
            comparison_df['Improvement_vs_Naive'] = (
                (naive_rmse - df['vol_rmse_original']) / naive_rmse * 100
            ).round(2)
        
        return comparison_df
    
    def plot_results(self):
        """Create visualization of results"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        models = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # Plot 1: RMSE comparison (standardized)
        ax = axes[0, 0]
        rmse_std = [self.results[m]['rmse_standardized'] for m in models]
        bars = ax.bar(models, rmse_std, color=colors)
        ax.set_ylabel('RMSE (Standardized)')
        ax.set_title('Overall RMSE Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, rmse_std):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.4f}', ha='center', va='bottom')
        
        # Plot 2: Volatility RMSE (original scale)
        ax = axes[0, 1]
        vol_rmse = [self.results[m]['vol_rmse_original'] for m in models]
        bars = ax.bar(models, vol_rmse, color=colors)
        ax.set_ylabel('RMSE (Original Scale)')
        ax.set_title('Volatility RMSE Comparison')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, vol_rmse):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.4f}', ha='center', va='bottom')
        
        # Plot 3: MAPE comparison
        ax = axes[0, 2]
        mape = [self.results[m]['vol_mape'] for m in models]
        bars = ax.bar(models, mape, color=colors)
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Mean Absolute Percentage Error')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, mape):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # Plot 4: R² Score
        ax = axes[1, 0]
        r2 = [self.results[m]['r2_standardized'] for m in models]
        bars = ax.bar(models, r2, color=colors)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.set_ylim([min(0, min(r2) - 0.1), 1])
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, r2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top')
        
        # Plot 5: Direction Accuracy
        ax = axes[1, 1]
        dir_acc = [self.results[m]['direction_accuracy'] * 100 for m in models]
        bars = ax.bar(models, dir_acc, color=colors)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Direction Prediction Accuracy')
        ax.set_ylim([0, 100])
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        for bar, val in zip(bars, dir_acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # Plot 6: Improvement vs Naive
        ax = axes[1, 2]
        if 'Naive' in self.results:
            naive_rmse = self.results['Naive']['vol_rmse_original']
            improvements = []
            model_names = []
            for m in models:
                if m != 'Naive':
                    imp = (naive_rmse - self.results[m]['vol_rmse_original']) / naive_rmse * 100
                    improvements.append(imp)
                    model_names.append(m)
            
            bars = ax.bar(model_names, improvements, 
                          color=[colors[models.index(m)] for m in model_names])
            ax.set_ylabel('Improvement (%)')
            ax.set_title('Improvement vs Naive Baseline')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            for bar, val in zip(bars, improvements):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}%', ha='center', 
                       va='bottom' if val >= 0 else 'top')
        
        plt.suptitle('Model Performance Comparison - SpotV2Net Research', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_dir = 'output/evaluation'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def save_results(self):
        """Save all evaluation results"""
        output_dir = 'output/evaluation'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save comparison table
        comparison_df = self.create_comparison_table()
        if comparison_df is not None:
            comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
            
            # Also save as LaTeX for paper
            latex_table = comparison_df.to_latex(float_format="%.4f")
            with open(os.path.join(output_dir, 'model_comparison.tex'), 'w') as f:
                f.write(latex_table)
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        output_dir = 'output/evaluation'
        
        report = []
        report.append("="*80)
        report.append("SPOTV2NET MODEL EVALUATION SUMMARY")
        report.append("="*80)
        report.append(f"\nEvaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Sequence Length: {self.seq_length}")
        report.append(f"Device: {self.device}")
        
        if self.results:
            report.append("\n" + "="*80)
            report.append("MODEL PERFORMANCE METRICS")
            report.append("="*80)
            
            # Find best model
            best_model = min(self.results.keys(), 
                           key=lambda x: self.results[x]['vol_rmse_original'])
            
            report.append(f"\n🏆 BEST MODEL: {best_model}")
            report.append(f"   Volatility RMSE (original): {self.results[best_model]['vol_rmse_original']:.6f}")
            
            # Detailed results for each model
            for model_name in sorted(self.results.keys()):
                metrics = self.results[model_name]
                report.append(f"\n{model_name} Model:")
                report.append(f"  Test Samples: {metrics['n_samples']}")
                report.append(f"  RMSE (standardized): {metrics['rmse_standardized']:.6f}")
                report.append(f"  Volatility RMSE (standardized): {metrics['vol_rmse_standardized']:.6f}")
                report.append(f"  Volatility RMSE (original): {metrics['vol_rmse_original']:.6f}")
                report.append(f"  Volatility MAE (original): {metrics['vol_mae_original']:.6f}")
                report.append(f"  Volatility MAPE: {metrics['vol_mape']:.2f}%")
                report.append(f"  R² Score: {metrics['r2_standardized']:.4f}")
                report.append(f"  Direction Accuracy: {metrics['direction_accuracy']*100:.2f}%")
            
            # Relative performance
            if 'Naive' in self.results:
                report.append("\n" + "="*80)
                report.append("RELATIVE PERFORMANCE vs NAIVE BASELINE")
                report.append("="*80)
                
                naive_rmse = self.results['Naive']['vol_rmse_original']
                for model_name in sorted(self.results.keys()):
                    if model_name != 'Naive':
                        model_rmse = self.results[model_name]['vol_rmse_original']
                        improvement = (naive_rmse - model_rmse) / naive_rmse * 100
                        report.append(f"{model_name}: {improvement:+.2f}% improvement")
        
        report.append("\n" + "="*80)
        report.append("KEY FINDINGS")
        report.append("="*80)
        
        # Analyze results
        if 'SpotV2Net' in self.results and 'LSTM' in self.results:
            gnn_rmse = self.results['SpotV2Net']['vol_rmse_original']
            lstm_rmse = self.results['LSTM']['vol_rmse_original']
            
            if gnn_rmse < lstm_rmse:
                improvement = (lstm_rmse - gnn_rmse) / lstm_rmse * 100
                report.append(f"✅ SpotV2Net outperforms LSTM by {improvement:.2f}%")
            else:
                deficit = (gnn_rmse - lstm_rmse) / lstm_rmse * 100
                report.append(f"⚠️  LSTM outperforms SpotV2Net by {deficit:.2f}%")
        
        # Save report
        report_text = "\n".join(report)
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION FOR SPOTV2NET RESEARCH")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(seq_length=42)
    
    # Evaluate all models
    print("\nEvaluating all models on the same test set...")
    
    # 1. Evaluate baselines first
    evaluator.evaluate_naive_baseline()
    evaluator.evaluate_mean_baseline()
    
    # 2. Evaluate LSTM
    lstm_dir = 'output/LSTM_42_fixed'
    if not os.path.exists(lstm_dir):
        # Try the enhanced version
        lstm_dir = 'output/LSTM_42'
    evaluator.evaluate_lstm(lstm_dir)
    
    # 3. Evaluate SpotV2Net
    gnn_dir = 'output/20240525_RGNN_std_optuna_42'
    evaluator.evaluate_gnn(gnn_dir)
    
    # Create comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    comparison_df = evaluator.create_comparison_table()
    if comparison_df is not None:
        print(comparison_df)
    
    # Plot results
    evaluator.plot_results()
    
    # Save all results
    evaluator.save_results()
    
    print("\n✅ Evaluation complete! Results saved to output/evaluation/")
    print("   - evaluation_results.json: Raw metrics")
    print("   - model_comparison.csv: Comparison table")
    print("   - model_comparison.png: Visualization")
    print("   - evaluation_report.txt: Summary report")


if __name__ == "__main__":
    main()