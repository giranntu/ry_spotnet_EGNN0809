#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for 30-Minute Intraday Volatility
=================================================================
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
            # Features shape: [batch, seq_length, features]
            # Target shape: [batch, 30]
            features = batch['features']
            target = batch['target']
            
            # Use last interval's volatilities as prediction
            # Extract last timestep volatilities (first 30 features)
            last_vols = features[:, -1, :30]
            
            all_preds.append(last_vols.numpy())
            all_targets.append(target.numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        # Calculate metrics using evaluator
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'Naive_30min',
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var']
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
        for i in range(min(1000, len(train_dataset))):  # Sample for efficiency
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
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var']
        }, y_pred
    
    def evaluate_har_intraday(self):
        """HAR model adapted for 30-minute intervals"""
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
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var']
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
        print(f"  Best Val QLIKE: {checkpoint.get('val_qlike', 'N/A'):.4f}")
        
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
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var']
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
        print(f"  Best Val QLIKE: {checkpoint.get('val_qlike', 'N/A'):.4f}")
        
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
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var']
        }, y_pred
    
    def create_comparison_table(self, metrics_list):
        """Create formatted comparison table"""
        df = pd.DataFrame(metrics_list)
        
        # Sort by QLIKE (primary metric for volatility forecasting)
        df = df.sort_values('qlike')
        
        # Format for display
        display_df = df.copy()
        
        # Format numbers
        for col in ['rmse', 'mae']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")
        
        for col in ['qlike']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        for col in ['rmse_var', 'mae_var']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.8f}")
        
        return display_df
    
    def plot_predictions_comparison(self, predictions_dict, n_samples=100, assets=[0, 5, 10, 15, 20]):
        """Plot predictions for selected assets"""
        n_assets = len(assets)
        fig, axes = plt.subplots(n_assets, 1, figsize=(14, 3*n_assets))
        if n_assets == 1:
            axes = [axes]
        
        colors = ['black', 'blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for idx, asset_idx in enumerate(assets):
            ax = axes[idx]
            
            # Get true values for this asset
            true_values = None
            
            for model_name, (metrics, preds) in predictions_dict.items():
                if preds is not None and len(preds) > 0:
                    if true_values is None:
                        # Extract true values from the first model's evaluation
                        # We need to get them from test loader
                        all_targets = []
                        for batch in self.test_loader:
                            all_targets.append(batch['target'].numpy())
                            if len(all_targets) * 32 >= n_samples:
                                break
                        true_values = np.vstack(all_targets)[:n_samples, asset_idx]
                        ax.plot(true_values, label='Actual', color='black', linewidth=2, alpha=0.8)
                    
                    # Plot predictions
                    model_preds = preds[:n_samples, asset_idx]
                    color_idx = list(predictions_dict.keys()).index(model_name) + 1
                    ax.plot(model_preds, label=model_name.replace('_30min', ''), 
                           alpha=0.7, color=colors[color_idx % len(colors)])
            
            ax.set_title(f'Asset {asset_idx + 1} - 30-Minute Volatility')
            ax.set_xlabel('Time (30-min intervals)')
            ax.set_ylabel('Volatility')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('30-Minute Intraday Volatility Predictions', fontsize=14)
        plt.tight_layout()
        plt.savefig('intraday_predictions_30min.png', dpi=120, bbox_inches='tight')
        plt.close()
        
        print("✅ Saved prediction plots to intraday_predictions_30min.png")
    
    def run_evaluation(self):
        """Run complete evaluation"""
        print("="*80)
        print("30-MINUTE INTRADAY VOLATILITY MODEL EVALUATION")
        print("="*80)
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Sequence length: {self.config['seq_length']} intervals (≈{self.config['seq_length']/13:.1f} days)")
        print(f"Forecast horizon: 1 interval (30 minutes)")
        
        # Check which models are available
        print("\nChecking available models:")
        lstm_available = os.path.exists(f'output/LSTM_30min_{self.config["seq_length"]}/best_model.pt')
        gnn_available = os.path.exists(f'output/{self.config["modelname"]}_30min_{self.config["seq_length"]}/best_model.pt')
        print(f"  LSTM: {'✓ Available' if lstm_available else '✗ Not found'}")
        print(f"  SpotV2Net: {'✓ Available' if gnn_available else '✗ Not found'}")
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
        
        har_metrics, har_pred = self.evaluate_har_intraday()
        if har_metrics:
            all_metrics.append(har_metrics)
            predictions['HAR'] = (har_metrics, har_pred)
        
        # Evaluate neural networks
        lstm_metrics, lstm_pred = self.evaluate_lstm()
        if lstm_metrics:
            all_metrics.append(lstm_metrics)
            predictions['LSTM'] = (lstm_metrics, lstm_pred)
        
        gnn_metrics, gnn_pred = self.evaluate_spotv2net()
        if gnn_metrics:
            all_metrics.append(gnn_metrics)
            predictions['SpotV2Net'] = (gnn_metrics, gnn_pred)
        
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
                    print(f"  {metrics['model']:20s}: {improvement:+.2f}%")
        
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
        
        output_file = f"evaluation_results_30min_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        
        return all_metrics


def main():
    evaluator = ComprehensiveEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()