#!/usr/bin/env python3
"""
Evaluation Script for Trained Models
=====================================
Loads and evaluates the actual trained SpotV2Net and LSTM models
"""

import numpy as np
import torch
import torch.nn as nn
import h5py
import pandas as pd
from pathlib import Path
import yaml
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from utils.models import GATModel
from torch_geometric.data import Data
import torch_geometric

# LSTM Model (same as in training script)
class LSTMModel(nn.Module):
    def __init__(self, input_size=900, hidden_size=256, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 900)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class ComprehensiveEvaluator:
    """Evaluator that loads actual trained models"""
    
    def __init__(self):
        # Load config
        with open('config/GNN_param.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Define splits
        self.seq_length = 42
        self.seq_offset = self.seq_length - 1
        self.train_end_idx = 967
        self.val_end_idx = 1219
        
        # Load test data
        self.load_test_data()
        
        # Device
        self.device = torch.device("cpu")
        
    def load_test_data(self):
        """Load test data"""
        print("Loading test data...")
        
        vols_file = 'processed_data/vols_mats_taq_standardized.h5'
        
        with h5py.File(vols_file, 'r') as f:
            sorted_keys = sorted(f.keys(), key=int)
            matrices = [f[key][:] for key in sorted_keys]
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(matrices) - self.seq_length):
            seq = np.array([matrices[j] for j in range(i, i+self.seq_length)])
            target = matrices[i+self.seq_length]
            sequences.append(seq)
            targets.append(target)
        
        # Extract test set
        self.X_test = np.array(sequences[self.val_end_idx:])
        self.y_test = np.array(targets[self.val_end_idx:])
        
        print(f"Test set: {len(self.X_test)} samples")
        
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """Calculate comprehensive metrics"""
        
        # Ensure numpy arrays
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Flatten for metrics
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # Core metrics
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # QLIKE metric
        epsilon = 1e-8
        y_true_safe = np.maximum(y_true_flat, epsilon)
        y_pred_safe = np.maximum(y_pred_flat, epsilon)
        
        qlike = np.mean(
            np.log(y_pred_safe**2 / y_true_safe**2) + 
            y_true_safe**2 / y_pred_safe**2 - 1
        )
        
        # R-squared
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Directional accuracy
        if len(y_true_flat) > 1:
            actual_changes = np.diff(y_true_flat)
            pred_changes = np.diff(y_pred_flat)
            mda = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
        else:
            mda = 0.0
        
        return {
            'model': model_name,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'qlike': float(qlike),
            'r2': float(r2),
            'mda': float(mda),
            'n_samples': len(y_true)
        }
    
    def evaluate_spotv2net(self):
        """Evaluate trained SpotV2Net model"""
        print("\n" + "="*80)
        print("Evaluating SpotV2Net Model")
        print("="*80)
        
        # Model path
        model_path = Path("output/20240525_RGNN_std_optuna_42")
        weights_file = model_path / "20240525_RGNN_std_optuna_weights_seed_5154.pth"
        
        if not weights_file.exists():
            print(f"SpotV2Net weights not found at {weights_file}")
            return None
        
        try:
            # Load model architecture
            model = GATModel(
                node_channels=30,
                edge_channels=1,
                hidden_channels=self.config['dim_hidden_layers'],
                out_channels=30,
                num_heads=self.config['num_heads'],
                concat_heads=self.config['concat_heads'],
                negative_slope=self.config['negative_slope'],
                dropout_gat=self.config['dropout_att'],
                dropout=self.config['dropout'],
                activation=self.config['activation']
            ).to(self.device)
            
            # Load weights
            checkpoint = torch.load(weights_file, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval()
            
            print(f"✅ Loaded SpotV2Net from {weights_file}")
            
            # Prepare test data as graphs
            predictions = []
            
            with torch.no_grad():
                for i in range(len(self.X_test)):
                    # Create graph data from sequence
                    seq = torch.FloatTensor(self.X_test[i])  # Shape: (42, 30, 30)
                    
                    # Use last matrix as node features
                    x = seq[-1].diagonal().unsqueeze(1)  # Shape: (30, 1)
                    
                    # Create fully connected edge index
                    edge_index = []
                    for src in range(30):
                        for dst in range(30):
                            if src != dst:
                                edge_index.append([src, dst])
                    edge_index = torch.LongTensor(edge_index).t()
                    
                    # Edge features from covariance
                    edge_attr = []
                    for src in range(30):
                        for dst in range(30):
                            if src != dst:
                                edge_attr.append([seq[-1, src, dst]])
                    edge_attr = torch.FloatTensor(edge_attr)
                    
                    # Create data object
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                    
                    # Get prediction
                    out = model(data)
                    
                    # Reconstruct matrix
                    pred_matrix = torch.zeros(30, 30)
                    pred_matrix.diagonal().copy_(out.squeeze())
                    predictions.append(pred_matrix.numpy())
            
            predictions = np.array(predictions)
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, predictions, "SpotV2Net")
            print(f"SpotV2Net RMSE: {metrics['rmse']:.6f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating SpotV2Net: {e}")
            return None
    
    def evaluate_lstm(self):
        """Evaluate trained LSTM model"""
        print("\n" + "="*80)
        print("Evaluating LSTM Model")
        print("="*80)
        
        # Try multiple possible paths
        possible_paths = [
            Path("output/LSTM_42_aligned/lstm_weights_seed_5154.pth"),
            Path("output/LSTM_42/best_lstm_weights.pth"),
            Path("output/LSTM_42/lstm_weights_seed_42.pth")
        ]
        
        weights_file = None
        for path in possible_paths:
            if path.exists():
                weights_file = path
                break
        
        if weights_file is None:
            print("No LSTM weights found")
            return None
        
        try:
            # Load model
            model = LSTMModel(input_size=900, hidden_size=256, num_layers=2).to(self.device)
            model.load_state_dict(torch.load(weights_file, map_location=self.device))
            model.eval()
            
            print(f"✅ Loaded LSTM from {weights_file}")
            
            # Prepare test data
            X_test_lstm = self.X_test.reshape(len(self.X_test), self.seq_length, -1)
            X_test_tensor = torch.FloatTensor(X_test_lstm)
            
            # Get predictions
            with torch.no_grad():
                predictions_flat = model(X_test_tensor)
                predictions = predictions_flat.reshape(-1, 30, 30)
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, predictions, "LSTM")
            print(f"LSTM RMSE: {metrics['rmse']:.6f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating LSTM: {e}")
            return None
    
    def evaluate_benchmarks(self):
        """Evaluate benchmark models"""
        print("\n" + "="*80)
        print("Evaluating Benchmark Models")
        print("="*80)
        
        results = []
        
        # Naive Persistence
        y_pred_naive = self.X_test[:, -1, :, :]
        naive_metrics = self.calculate_metrics(self.y_test, y_pred_naive, "Naive_Persistence")
        print(f"Naive RMSE: {naive_metrics['rmse']:.6f}")
        results.append(naive_metrics)
        
        # HAR Model
        lookbacks = [1, 5, 22]
        weights = [1/3, 1/3, 1/3]
        y_pred_har = np.zeros_like(self.y_test)
        
        for lookback, weight in zip(lookbacks, weights):
            if lookback <= self.seq_length:
                component = np.mean(self.X_test[:, -lookback:, :, :], axis=1)
                y_pred_har += weight * component
        
        har_metrics = self.calculate_metrics(self.y_test, y_pred_har, "HAR_Model")
        print(f"HAR RMSE: {har_metrics['rmse']:.6f}")
        results.append(har_metrics)
        
        return results
    
    def run_complete_evaluation(self):
        """Run full evaluation"""
        
        print("="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        print(f"Test samples: {len(self.y_test)}")
        print(f"Test period: 2024-2025")
        print("="*80)
        
        all_results = []
        
        # Benchmarks
        benchmark_results = self.evaluate_benchmarks()
        all_results.extend(benchmark_results)
        
        # Neural models
        spotv2_metrics = self.evaluate_spotv2net()
        if spotv2_metrics:
            all_results.append(spotv2_metrics)
        
        lstm_metrics = self.evaluate_lstm()
        if lstm_metrics:
            all_results.append(lstm_metrics)
        
        # Create comparison table
        print("\n" + "="*80)
        print("FINAL PERFORMANCE COMPARISON")
        print("="*80)
        
        df = pd.DataFrame(all_results)
        df = df.sort_values('rmse')
        
        # Format for display
        display_cols = ['model', 'rmse', 'mae', 'r2', 'mda', 'qlike']
        print("\n", df[display_cols].to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'test_samples': len(self.y_test),
            'metrics': all_results,
            'best_model': df.iloc[0]['model'],
            'best_rmse': float(df.iloc[0]['rmse'])
        }
        
        output_file = f"final_evaluation_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        print(f"🏆 Best Model: {df.iloc[0]['model']}")
        print(f"📊 Best RMSE: {df.iloc[0]['rmse']:.6f}")
        
        if len(df) > 1:
            improvement = (df.iloc[-1]['rmse'] - df.iloc[0]['rmse']) / df.iloc[-1]['rmse'] * 100
            print(f"📈 Improvement over Naive: {improvement:.2f}%")
        
        return all_results


def main():
    evaluator = ComprehensiveEvaluator()
    evaluator.run_complete_evaluation()


if __name__ == "__main__":
    main()