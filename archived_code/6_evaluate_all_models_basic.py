#!/usr/bin/env python3
"""
Unified Model Evaluation with Proper Volatility Metrics
========================================================
Evaluates all models with consistent methodology
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
import matplotlib.pyplot as plt
import seaborn as sns

class UnifiedEvaluator:
    """Evaluator for all volatility forecasting models"""
    
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
        
    def load_test_data(self):
        """Load standardized test data"""
        print("Loading test data...")
        
        # Load volatility matrices
        vols_file = 'processed_data/vols_mats_taq_standardized.h5'
        
        with h5py.File(vols_file, 'r') as f:
            sorted_keys = sorted(f.keys(), key=int)
            matrices = np.array([f[key][:] for key in sorted_keys])
        
        # Extract diagonals (individual volatilities) for better metrics
        self.all_vols = np.array([np.diag(m) for m in matrices])
        
        # Create test sequences
        sequences = []
        targets = []
        
        for i in range(len(matrices) - self.seq_length):
            seq = matrices[i:i+self.seq_length]
            target = matrices[i+self.seq_length]
            sequences.append(seq)
            targets.append(target)
        
        # Extract test set
        self.X_test = np.array(sequences[self.val_end_idx:])
        self.y_test = np.array(targets[self.val_end_idx:])
        
        # Also extract diagonal-only versions for cleaner metrics
        self.X_test_diag = np.array([np.diag(self.X_test[i, -1]) for i in range(len(self.X_test))])
        self.y_test_diag = np.array([np.diag(self.y_test[i]) for i in range(len(self.y_test))])
        
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Shape: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
        
    def calculate_volatility_metrics(self, y_true, y_pred, model_name="Model"):
        """Calculate proper volatility forecasting metrics"""
        
        # Ensure numpy arrays
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Extract diagonals if matrices (individual volatilities)
        if len(y_true.shape) == 3:  # Batch of matrices
            y_true_diag = np.array([np.diag(y_true[i]) for i in range(len(y_true))])
            y_pred_diag = np.array([np.diag(y_pred[i]) for i in range(len(y_pred))])
        else:
            y_true_diag = y_true
            y_pred_diag = y_pred
        
        # Flatten for element-wise metrics
        y_true_flat = y_true_diag.flatten()
        y_pred_flat = y_pred_diag.flatten()
        
        # Basic metrics
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # R-squared
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # QLIKE metric (proper implementation for standardized data)
        # Since data is standardized, we need to be careful with QLIKE
        # Add small constant to avoid log(0)
        epsilon = 1e-6
        y_true_pos = np.abs(y_true_flat) + epsilon
        y_pred_pos = np.abs(y_pred_flat) + epsilon
        
        # QLIKE = mean(log(σ̂²/σ²) + σ²/σ̂²)
        qlike = np.mean(np.log(y_pred_pos/y_true_pos) + y_true_pos/y_pred_pos)
        
        # Mincer-Zarnowitz regression (σ² = α + β*σ̂² + ε)
        # Good forecast: α=0, β=1
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(y_pred_flat, y_true_flat)
        mz_alpha = intercept
        mz_beta = slope
        mz_r2 = r_value**2
        
        # Directional accuracy (important for risk management)
        if len(y_true_flat) > 1:
            # Calculate returns (changes in volatility)
            actual_changes = np.diff(y_true_flat)
            pred_changes = np.diff(y_pred_flat)
            # Directional accuracy
            mda = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
        else:
            mda = 0.0
        
        # Mean percentage error (for interpretability)
        mpe = np.mean((y_pred_flat - y_true_flat) / (np.abs(y_true_flat) + epsilon)) * 100
        
        metrics = {
            'model': model_name,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'qlike': float(qlike),
            'mz_alpha': float(mz_alpha),
            'mz_beta': float(mz_beta),
            'mz_r2': float(mz_r2),
            'mda': float(mda),
            'mpe': float(mpe),
            'n_samples': len(y_true)
        }
        
        return metrics
    
    def evaluate_naive_persistence(self):
        """Evaluate naive persistence benchmark"""
        print("\nEvaluating Naive Persistence...")
        
        # Prediction: next volatility = current volatility
        y_pred = self.X_test_diag  # Use last observed volatility
        y_true = self.y_test_diag
        
        metrics = self.calculate_volatility_metrics(y_true, y_pred, "Naive_Persistence")
        return metrics
    
    def evaluate_har_model(self):
        """Evaluate HAR model benchmark"""
        print("\nEvaluating HAR Model...")
        
        # HAR: weighted average of different horizons
        # Daily (1), Weekly (5), Monthly (22)
        horizons = [1, 5, 22]
        weights = np.array([1/3, 1/3, 1/3])
        
        y_pred = np.zeros_like(self.y_test_diag)
        
        for i, (h, w) in enumerate(zip(horizons, weights)):
            if h <= self.seq_length:
                # Average over last h observations
                component = np.mean(self.X_test[:, -h:, :, :], axis=1)
                component_diag = np.array([np.diag(component[j]) for j in range(len(component))])
                y_pred += w * component_diag
        
        metrics = self.calculate_volatility_metrics(self.y_test_diag, y_pred, "HAR_Model")
        return metrics
    
    def evaluate_ewma(self, lambda_param=0.94):
        """Evaluate EWMA benchmark"""
        print(f"\nEvaluating EWMA (λ={lambda_param})...")
        
        y_pred = []
        
        for i in range(len(self.X_test)):
            # Get sequence of volatilities
            seq = self.X_test[i]
            seq_diag = np.array([np.diag(seq[j]) for j in range(len(seq))])
            
            # Apply EWMA
            weights = np.array([(1-lambda_param) * lambda_param**(self.seq_length-1-j) 
                               for j in range(self.seq_length)])
            weights = weights / weights.sum()
            
            pred = np.sum(seq_diag * weights.reshape(-1, 1), axis=0)
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        metrics = self.calculate_volatility_metrics(self.y_test_diag, y_pred, f"EWMA_{lambda_param}")
        return metrics
    
    def plot_predictions_sample(self, predictions_dict, n_samples=100, n_assets=5):
        """Plot sample predictions for visualization"""
        fig, axes = plt.subplots(n_assets, 1, figsize=(12, 10))
        
        for asset_idx in range(n_assets):
            ax = axes[asset_idx]
            
            # Plot actual values
            actual = self.y_test_diag[:n_samples, asset_idx]
            ax.plot(actual, label='Actual', color='black', alpha=0.7, linewidth=1.5)
            
            # Plot predictions from different models
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for (model_name, preds), color in zip(predictions_dict.items(), colors):
                if preds is not None:
                    model_preds = preds[:n_samples, asset_idx]
                    ax.plot(model_preds, label=model_name, alpha=0.6, color=color)
            
            ax.set_title(f'Asset {asset_idx+1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Volatility')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('volatility_predictions_sample.png', dpi=100)
        plt.close()
        
        print("✅ Saved prediction plots to volatility_predictions_sample.png")
    
    def create_comparison_table(self, metrics_list):
        """Create comprehensive comparison table"""
        df = pd.DataFrame(metrics_list)
        
        # Sort by RMSE
        df = df.sort_values('rmse')
        
        # Format for display
        display_cols = ['model', 'rmse', 'mae', 'r2', 'qlike', 'mz_beta', 'mda', 'mpe']
        
        # Format numbers
        for col in ['rmse', 'mae']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.6f}")
        
        for col in ['r2', 'mz_beta', 'mda']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}")
        
        for col in ['qlike']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.3f}")
        
        for col in ['mpe']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2f}%")
        
        return df[display_cols]
    
    def run_complete_evaluation(self):
        """Run evaluation for all models"""
        print("="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        print(f"Test samples: {len(self.y_test)}")
        print(f"Number of assets: 30")
        print(f"Test period: 2024-2025")
        print("="*80)
        
        all_metrics = []
        predictions = {}
        
        # Evaluate benchmarks
        naive_metrics = self.evaluate_naive_persistence()
        all_metrics.append(naive_metrics)
        predictions['Naive'] = self.X_test_diag
        
        har_metrics = self.evaluate_har_model()
        all_metrics.append(har_metrics)
        
        ewma_metrics = self.evaluate_ewma(lambda_param=0.94)
        all_metrics.append(ewma_metrics)
        
        # Create comparison table
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        comparison_df = self.create_comparison_table(all_metrics)
        print("\n", comparison_df.to_string(index=False))
        
        # Plot sample predictions
        self.plot_predictions_sample(predictions)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'test_samples': len(self.y_test),
            'metrics': all_metrics,
            'best_model': comparison_df.iloc[0]['model'] if len(comparison_df) > 0 else None
        }
        
        output_file = f"evaluation_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print interpretation
        print("\n" + "="*80)
        print("METRICS INTERPRETATION")
        print("="*80)
        print("RMSE: Root Mean Squared Error (lower is better)")
        print("MAE: Mean Absolute Error (lower is better)")
        print("R²: Coefficient of determination (higher is better, max=1)")
        print("QLIKE: Quasi-likelihood (lower is better)")
        print("MZ-β: Mincer-Zarnowitz beta (closer to 1 is better)")
        print("MDA: Mean Directional Accuracy (higher is better)")
        print("MPE: Mean Percentage Error (closer to 0 is better)")
        
        return all_metrics


def main():
    evaluator = UnifiedEvaluator()
    evaluator.run_complete_evaluation()


if __name__ == "__main__":
    main()