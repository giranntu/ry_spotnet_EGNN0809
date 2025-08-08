#!/usr/bin/env python3
"""
Proper Evaluation Metrics with Inverse Transformations
=======================================================
Ensures all losses are calculated in the REAL volatility scale

CRITICAL: 
- Model outputs are in standardized log-space
- Must inverse transform before calculating ANY losses
- QLIKE has specific economic meaning only in real scale
"""

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class VolatilityEvaluator:
    """
    Evaluates volatility predictions with proper scale transformations
    
    CRITICAL FEATURES:
    1. Handles log-transformed standardized predictions
    2. Properly inverts transformations before loss calculation
    3. Implements QLIKE loss with economic interpretation
    """
    
    def __init__(self, scaler_params_file: str = None):
        """
        Initialize evaluator
        
        Args:
            scaler_params_file: Path to CSV with standardization parameters
        """
        self.scaler_params = None
        
        if scaler_params_file:
            self.load_scaler_params(scaler_params_file)
    
    def load_scaler_params(self, filepath: str):
        """
        Load standardization parameters from training
        """
        params_df = pd.read_csv(filepath)
        
        # Extract parameters based on method
        if 'Vol_LogMean' in params_df.columns:
            # Log-transform method
            self.scaler_params = {
                'method': 'log_transform',
                'log_mean': params_df['Vol_LogMean'].iloc[0],
                'log_std': params_df['Vol_LogStd'].iloc[0],
                'orig_mean': params_df['Vol_OrigMean'].iloc[0],
                'orig_std': params_df['Vol_OrigStd'].iloc[0]
            }
        elif 'Vol_Mean' in params_df.columns:
            # Scale-only method
            self.scaler_params = {
                'method': 'scale_only',
                'mean': params_df['Vol_Mean'].iloc[0],
                'std': params_df['Vol_Std'].iloc[0]
            }
        else:
            raise ValueError("Unknown scaler parameters format")
        
        print(f"Loaded {self.scaler_params['method']} scaler parameters")
    
    def inverse_transform(self, 
                         y_pred_scaled: np.ndarray, 
                         y_true_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse transform predictions and targets to real volatility scale
        
        CRITICAL IMPLEMENTATION:
        1. Inverse standardization (multiply by std, add mean)
        2. Inverse log transform (exp)
        3. Ensure positivity
        
        Args:
            y_pred_scaled: Model predictions in standardized log-space
            y_true_scaled: True values in standardized log-space
            
        Returns:
            (y_pred_real, y_true_real) in actual volatility units
        """
        if self.scaler_params is None:
            raise ValueError("Scaler parameters not loaded")
        
        method = self.scaler_params['method']
        
        if method == 'log_transform':
            # Step 1: Inverse standardization (z-score to log-space)
            y_pred_log = (y_pred_scaled * self.scaler_params['log_std'] + 
                         self.scaler_params['log_mean'])
            y_true_log = (y_true_scaled * self.scaler_params['log_std'] + 
                         self.scaler_params['log_mean'])
            
            # Step 2: Inverse log transform (log-space to real volatility)
            y_pred_real = np.exp(y_pred_log)
            y_true_real = np.exp(y_true_log)
            
        elif method == 'scale_only':
            # Only scaled by std, no log transform
            y_pred_real = y_pred_scaled * self.scaler_params['std']
            y_true_real = y_true_scaled * self.scaler_params['std']
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Step 3: Ensure positivity (volatility must be positive)
        y_pred_real = np.maximum(y_pred_real, 1e-8)
        y_true_real = np.maximum(y_true_real, 1e-8)
        
        return y_pred_real, y_true_real
    
    def calculate_mse(self, y_pred_real: np.ndarray, y_true_real: np.ndarray) -> float:
        """
        Mean Squared Error in real volatility scale
        """
        return np.mean((y_pred_real - y_true_real) ** 2)
    
    def calculate_rmse(self, y_pred_real: np.ndarray, y_true_real: np.ndarray) -> float:
        """
        Root Mean Squared Error in real volatility scale
        """
        return np.sqrt(self.calculate_mse(y_pred_real, y_true_real))
    
    def calculate_mae(self, y_pred_real: np.ndarray, y_true_real: np.ndarray) -> float:
        """
        Mean Absolute Error in real volatility scale
        """
        return np.mean(np.abs(y_pred_real - y_true_real))
    
    def calculate_qlike(self, var_pred_real: np.ndarray, var_true_real: np.ndarray) -> float:
        """
        Quasi-Likelihood Loss (QLIKE) for variance forecasting
        
        Standard QLIKE definition: L(σ², σ̂²) = σ̂²/σ² - ln(σ̂²/σ²) - 1
        
        CRITICAL NOTES:
        1. QLIKE is defined on VARIANCE (σ²), not volatility (σ)
        2. Inputs should already be variances (V_YZ values)
        3. Do NOT square the inputs - they are already variances!
        
        This loss function has important properties:
        1. Asymmetric penalty (under-prediction more costly than over-prediction)
        2. Scale-free (percentage errors matter)
        3. Consistent ranking with economic loss functions
        
        Args:
            var_pred_real: Predicted VARIANCE in real scale
            var_true_real: True VARIANCE in real scale
        """
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        var_pred_real = np.maximum(var_pred_real, epsilon)
        var_true_real = np.maximum(var_true_real, epsilon)
        
        # Calculate standard QLIKE formula
        ratio = var_pred_real / var_true_real
        qlike = np.mean(ratio - np.log(ratio) - 1)
        
        return qlike
    
    def calculate_all_metrics(self, 
                             y_pred_scaled: np.ndarray, 
                             y_true_scaled: np.ndarray,
                             is_variance: bool = True) -> Dict[str, float]:
        """
        Calculate all evaluation metrics with proper transformations
        
        Args:
            y_pred_scaled: Model predictions (standardized log-space)
            y_true_scaled: True values (standardized log-space)
            is_variance: True if predictions are variances, False if volatilities
            
        Returns:
            Dictionary of metrics in real scale
        """
        # CRITICAL: Transform to real scale first!
        y_pred_real, y_true_real = self.inverse_transform(y_pred_scaled, y_true_scaled)
        
        # These are VARIANCES after inverse transform
        var_pred_real = y_pred_real
        var_true_real = y_true_real
        
        # For RMSE/MAE, we report in volatility scale (more interpretable)
        vol_pred_real = np.sqrt(var_pred_real)
        vol_true_real = np.sqrt(var_true_real)
        
        metrics = {
            # Volatility-based metrics (economically interpretable)
            'rmse_vol': self.calculate_rmse(vol_pred_real, vol_true_real),
            'mae_vol': self.calculate_mae(vol_pred_real, vol_true_real),
            
            # Variance-based metrics
            'rmse_var': self.calculate_rmse(var_pred_real, var_true_real),
            'mae_var': self.calculate_mae(var_pred_real, var_true_real),
            
            # QLIKE uses variance directly
            'qlike': self.calculate_qlike(var_pred_real, var_true_real),
            
            # Also calculate metrics in scaled space for comparison
            'rmse_scaled': np.sqrt(np.mean((y_pred_scaled - y_true_scaled) ** 2)),
            'mae_scaled': np.mean(np.abs(y_pred_scaled - y_true_scaled)),
            
            # Statistics about predictions (in volatility scale)
            'pred_vol_mean': np.mean(vol_pred_real),
            'pred_vol_std': np.std(vol_pred_real),
            'true_vol_mean': np.mean(vol_true_real),
            'true_vol_std': np.std(vol_true_real),
        }
        
        return metrics
    
    def evaluate_by_interval(self,
                            y_pred_scaled: np.ndarray,
                            y_true_scaled: np.ndarray,
                            interval_labels: np.ndarray) -> pd.DataFrame:
        """
        Evaluate performance by time of day (which 30-min interval)
        
        This can reveal if the model performs better at certain times
        (e.g., market open vs close)
        """
        # Transform to real scale
        y_pred_real, y_true_real = self.inverse_transform(y_pred_scaled, y_true_scaled)
        
        results = []
        
        for interval in range(13):  # 13 intervals per day
            mask = interval_labels == interval
            
            if np.sum(mask) > 0:
                metrics = {
                    'interval': interval,
                    'time': f"{9.5 + interval*0.5:.1f}:00",  # Convert to clock time
                    'n_samples': np.sum(mask),
                    'rmse': self.calculate_rmse(y_pred_real[mask], y_true_real[mask]),
                    'mae': self.calculate_mae(y_pred_real[mask], y_true_real[mask]),
                    'qlike': self.calculate_qlike(y_pred_real[mask], y_true_real[mask])
                }
                results.append(metrics)
        
        return pd.DataFrame(results)


def compare_with_baseline(y_pred_scaled: np.ndarray,
                         y_true_scaled: np.ndarray,
                         evaluator: VolatilityEvaluator) -> pd.DataFrame:
    """
    Compare model predictions with simple baselines
    
    Baselines:
    1. Naive: Use previous value as prediction
    2. Historical mean: Use training set mean
    3. EWMA: Exponentially weighted moving average
    """
    # Transform to real scale
    y_pred_real, y_true_real = evaluator.inverse_transform(y_pred_scaled, y_true_scaled)
    
    results = []
    
    # Model performance
    model_metrics = evaluator.calculate_all_metrics(y_pred_scaled, y_true_scaled)
    results.append({
        'model': 'GNN/LSTM',
        'rmse': model_metrics['rmse'],
        'mae': model_metrics['mae'],
        'qlike': model_metrics['qlike']
    })
    
    # Naive baseline (previous value)
    y_naive = np.roll(y_true_real, 1)
    y_naive[0] = y_true_real[0]  # First prediction equals first true
    
    results.append({
        'model': 'Naive',
        'rmse': evaluator.calculate_rmse(y_naive, y_true_real),
        'mae': evaluator.calculate_mae(y_naive, y_true_real),
        'qlike': evaluator.calculate_qlike(y_naive, y_true_real)
    })
    
    # Historical mean baseline
    y_mean = np.full_like(y_true_real, np.mean(y_true_real))
    
    results.append({
        'model': 'Historical Mean',
        'rmse': evaluator.calculate_rmse(y_mean, y_true_real),
        'mae': evaluator.calculate_mae(y_mean, y_true_real),
        'qlike': evaluator.calculate_qlike(y_mean, y_true_real)
    })
    
    # EWMA baseline
    alpha = 0.94  # Common for volatility (RiskMetrics)
    y_ewma = np.zeros_like(y_true_real)
    y_ewma[0] = y_true_real[0]
    
    for t in range(1, len(y_true_real)):
        y_ewma[t] = alpha * y_ewma[t-1] + (1-alpha) * y_true_real[t-1]
    
    results.append({
        'model': 'EWMA (α=0.94)',
        'rmse': evaluator.calculate_rmse(y_ewma, y_true_real),
        'mae': evaluator.calculate_mae(y_ewma, y_true_real),
        'qlike': evaluator.calculate_qlike(y_ewma, y_true_real)
    })
    
    return pd.DataFrame(results)


def main():
    """
    Example evaluation with proper transformations
    """
    print("="*80)
    print("VOLATILITY EVALUATION WITH PROPER INVERSE TRANSFORMATIONS")
    print("="*80)
    
    # Example: Create dummy predictions for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate standardized log-space predictions (what model outputs)
    y_true_scaled = np.random.randn(n_samples) * 0.5  # Standardized
    y_pred_scaled = y_true_scaled + np.random.randn(n_samples) * 0.2  # Add noise
    
    # Create evaluator with mock parameters
    evaluator = VolatilityEvaluator()
    evaluator.scaler_params = {
        'method': 'log_transform',
        'log_mean': -2.5,  # log(volatility) typically negative
        'log_std': 0.5,
        'orig_mean': 0.15,  # 15% average volatility
        'orig_std': 0.08    # 8% std of volatility
    }
    
    # Calculate metrics
    metrics = evaluator.calculate_all_metrics(y_pred_scaled, y_true_scaled)
    
    print("\n📊 Evaluation Results:")
    print("-"*40)
    print("Metrics in REAL volatility scale:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.6f}")
    print("\nMetrics in scaled space (for comparison):")
    print(f"  RMSE (scaled): {metrics['rmse_scaled']:.6f}")
    print(f"  MAE (scaled): {metrics['mae_scaled']:.6f}")
    print("\nPrediction statistics (real scale):")
    print(f"  Pred mean: {metrics['pred_mean_real']:.4f}")
    print(f"  Pred std: {metrics['pred_std_real']:.4f}")
    print(f"  True mean: {metrics['true_mean_real']:.4f}")
    print(f"  True std: {metrics['true_std_real']:.4f}")
    
    # Compare with baselines
    print("\n📈 Baseline Comparison:")
    print("-"*40)
    baseline_df = compare_with_baseline(y_pred_scaled, y_true_scaled, evaluator)
    print(baseline_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ KEY INSIGHTS:")
    print("1. RMSE in real scale is MUCH larger than in scaled space")
    print("2. This is the TRUE prediction error in volatility units")
    print("3. QLIKE provides asymmetric loss (under-prediction more costly)")
    print("4. Model should beat naive baseline to be useful")
    print("="*80)


if __name__ == "__main__":
    main()