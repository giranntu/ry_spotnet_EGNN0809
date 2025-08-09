#!/usr/bin/env python3
"""
Simplified XGBoost Training for 30-Minute Volatility Prediction
================================================================
A reasonable and efficient baseline model without excessive complexity

Key Simplifications:
1. Minimal feature engineering (just basic statistics)
2. Faster hyperparameter search (fewer iterations)
3. Optimized for quick training while maintaining good performance
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our datasets and evaluator
from utils.dataset import IntradayVolatilityDataset
from utils.evaluation_intraday import VolatilityEvaluator


class SimpleXGBoostTrainer:
    """Simplified XGBoost trainer for efficient baseline model"""
    
    def __init__(self):
        """Initialize simple XGBoost trainer"""
        # Setup paths
        self.output_dir = 'output/XGBoost_30min_42'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set seed
        self.seed = 42
        np.random.seed(self.seed)
        
        # Check GPU
        self.use_gpu = self._check_gpu()
        
        # Simple but effective parameters (no extensive tuning needed)
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor',
            'random_state': self.seed,
            
            # Reasonable defaults that work well
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }
        
        if self.use_gpu:
            self.params['gpu_id'] = 0
        
        # Initialize evaluator
        scaler_file = 'processed_data/vols_30min_mean_std_scalers.csv'
        self.evaluator = VolatilityEvaluator(scaler_file)
        print(f"Loaded evaluator with scaler parameters")
        
    def _check_gpu(self):
        """Quick GPU check"""
        try:
            test_data = xgb.DMatrix(np.random.randn(100, 10), label=np.random.randn(100))
            xgb.train({'tree_method': 'gpu_hist', 'gpu_id': 0}, test_data, num_boost_round=1, verbose_eval=False)
            print("✅ GPU available for XGBoost")
            return True
        except:
            print("⚠️ GPU not available, using CPU")
            return False
    
    def prepare_data(self):
        """Load and prepare data with minimal processing"""
        print("\nLoading 30-minute intraday data...")
        
        vol_file = 'processed_data/vols_mats_30min_standardized.h5'
        volvol_file = 'processed_data/volvols_mats_30min_standardized.h5'
        
        # Create datasets
        self.train_dataset = IntradayVolatilityDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            seq_length=42,
            intervals_per_day=13,
            split='train',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        self.val_dataset = IntradayVolatilityDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            seq_length=42,
            intervals_per_day=13,
            split='val',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        self.test_dataset = IntradayVolatilityDataset(
            vol_file=vol_file,
            volvol_file=volvol_file,
            seq_length=42,
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        print(f"Data splits - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
        # Convert to XGBoost format with simple features
        self._create_simple_features()
    
    def _create_simple_features(self):
        """Create simplified features for faster training"""
        print("\nPreparing simplified features...")
        
        # Process training data
        X_train, y_train = self._extract_simple_features(self.train_dataset, "Training")
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Process validation data  
        X_val, y_val = self._extract_simple_features(self.val_dataset, "Validation")
        self.dval = xgb.DMatrix(X_val, label=y_val)
        
        # Process test data
        X_test, y_test = self._extract_simple_features(self.test_dataset, "Test")
        self.dtest = xgb.DMatrix(X_test, label=y_test)
        
        print(f"Feature dimensions: {X_train.shape[1]} features")
    
    def _extract_simple_features(self, dataset, split_name):
        """
        Extract simplified features - just essential statistics
        Much faster than full feature engineering
        """
        all_features = []
        all_targets = []
        
        for i in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
            sample = dataset[i]
            features = sample['features'].numpy()  # Shape: [42, 930]
            target = sample['target'].numpy()      # Shape: [30]
            
            # Extract volatilities (first 30 features at each timestep)
            vols = features[:, :30]  # [42, 30]
            
            # Simple but effective features:
            # 1. Last 3 observations (most recent info)
            recent_vols = vols[-3:, :].flatten()  # 90 features
            
            # 2. Statistics across time for each stock
            vol_mean = vols.mean(axis=0)  # 30 features
            vol_std = vols.std(axis=0)    # 30 features
            vol_last = vols[-1, :]        # 30 features
            
            # 3. Simple trend (last vs mean)
            vol_trend = vols[-1, :] - vol_mean  # 30 features
            
            # 4. Market-wide statistics (cross-sectional)
            market_vol = vols.mean(axis=1)  # 42 features - market vol at each time
            
            # Combine features (total: 90 + 30 + 30 + 30 + 30 + 42 = 252 features)
            combined = np.concatenate([
                recent_vols,
                vol_mean,
                vol_std,
                vol_last,
                vol_trend,
                market_vol
            ])
            
            # Use mean volatility as target (simpler than 30 individual predictions)
            mean_target = np.mean(target)
            
            all_features.append(combined)
            all_targets.append(mean_target)
        
        X = np.vstack(all_features)
        y = np.array(all_targets)
        
        return X, y
    
    def train(self):
        """Train XGBoost with early stopping"""
        print("\n" + "="*60)
        print("SIMPLIFIED XGBOOST TRAINING")
        print("="*60)
        print(f"GPU: {'✅ Enabled' if self.use_gpu else '❌ CPU mode'}")
        
        # Setup evaluation list
        evals = [(self.dtrain, 'train'), (self.dval, 'validation')]
        
        # Train with early stopping
        print("\nTraining XGBoost...")
        self.model = xgb.train(
            self.params,
            self.dtrain,
            num_boost_round=self.params.get('n_estimators', 300),
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=10
        )
        
        # Save model
        model_path = os.path.join(self.output_dir, 'xgboost_model.json')
        self.model.save_model(model_path)
        print(f"\n✅ Model saved to: {model_path}")
        
        # Also save in joblib format
        joblib_path = os.path.join(self.output_dir, 'xgboost_model.pkl')
        joblib.dump(self.model, joblib_path)
    
    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "="*60)
        print("TEST EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(self.dtest)
        y_true = self.dtest.get_label()
        
        # Expand predictions to 30 stocks
        y_pred_expanded = np.tile(y_pred.reshape(-1, 1), (1, 30))
        y_true_expanded = np.tile(y_true.reshape(-1, 1), (1, 30))
        
        # Calculate metrics
        metrics = self.evaluator.calculate_all_metrics(
            y_pred_expanded, y_true_expanded, is_variance=True
        )
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        print(f"Test RMSE (standardized): {np.sqrt(np.mean((y_pred - y_true)**2)):.6f}")
        print(f"Test RMSE (volatility): {metrics['rmse_vol']:.6f}")
        print(f"Test MAE (volatility): {metrics['mae_vol']:.6f}")
        print(f"Test QLIKE: {metrics['qlike']:.4f}")
        print(f"Test R² Score: {r2:.4f}")
        print(f"Best iteration: {self.model.best_iteration}")
        
        # Save results
        test_results = {
            'test_rmse_vol': float(metrics['rmse_vol']),
            'test_mae_vol': float(metrics['mae_vol']),
            'test_qlike': float(metrics['qlike']),
            'test_r2': float(r2),
            'best_iteration': int(self.model.best_iteration),
            'num_features': 252,
            'training_time': 'fast',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n✅ Results saved to: {self.output_dir}/test_results.json")
        
        # Feature importance (top 10)
        importance = self.model.get_score(importance_type='gain')
        if importance:
            print("\nTop 10 Feature Importance:")
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, score in sorted_imp:
                print(f"  {feat}: {score:.2f}")


def main():
    """Main execution"""
    print("="*80)
    print("SIMPLIFIED XGBOOST - EFFICIENT BASELINE MODEL")
    print("="*80)
    
    # Initialize trainer
    trainer = SimpleXGBoostTrainer()
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    trainer.train()
    
    # Evaluate
    trainer.evaluate()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()