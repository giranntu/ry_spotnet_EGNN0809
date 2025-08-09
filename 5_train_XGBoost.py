#!/usr/bin/env python3
"""
XGBoost Training for 30-Minute Intraday Volatility Prediction
==============================================================
GPU-accelerated XGBoost implementation aligned with existing pipeline

Key Features:
1. Uses same IntradayVolatilityDataset as LSTM/GNN
2. GPU acceleration via 'gpu_hist' tree method
3. Proper evaluation with VolatilityEvaluator
4. Temporal train/val/test splits (60/20/20)
5. Feature engineering optimized for tree-based models
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import json
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our datasets and evaluator
from utils.dataset import IntradayVolatilityDataset
from utils.evaluation_intraday import VolatilityEvaluator


class XGBoostVolatilityTrainer:
    """XGBoost trainer for 30-minute volatility prediction with GPU support"""
    
    def __init__(self, config_path='config/GNN_param.yaml'):
        """Initialize XGBoost trainer with configuration"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Update for 30-minute data
        self.config['seq_length'] = 42  # 42 thirty-minute intervals
        
        # Setup paths
        self.output_dir = f'output/XGBoost_30min_{self.config["seq_length"]}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training parameters
        self.seed = self.config['seed'][0] if isinstance(self.config['seed'], list) else self.config['seed']
        np.random.seed(self.seed)
        
        # Check GPU availability
        self.use_gpu = self._check_gpu_availability()
        
        # XGBoost parameters optimized for volatility forecasting
        self.base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'predictor': 'gpu_predictor' if self.use_gpu else 'cpu_predictor',
            'random_state': self.seed,
            'n_jobs': -1 if not self.use_gpu else 1,
            
            # Hyperparameters to tune
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }
        
        if self.use_gpu:
            self.base_params['gpu_id'] = 0
        
        # Initialize evaluator
        scaler_file = 'processed_data/vols_30min_mean_std_scalers.csv'
        if not os.path.exists(scaler_file):
            raise FileNotFoundError(
                f"CRITICAL: Scaler file not found at {scaler_file}. "
                f"Please run standardization (script 4) first!"
            )
        self.evaluator = VolatilityEvaluator(scaler_file)
        print(f"Loaded evaluator with scaler parameters from {scaler_file}")
        
        # Metrics tracking
        self.train_history = {'rmse': [], 'qlike': []}
        self.val_history = {'rmse': [], 'qlike': []}
        
    def _check_gpu_availability(self):
        """Check if GPU is available for XGBoost"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"GPU detected: {gpu.name}")
                print(f"GPU Memory: {gpu.memoryTotal:.0f} MB")
                print(f"GPU Memory Free: {gpu.memoryFree:.0f} MB")
                return True
        except ImportError:
            print("GPUtil not installed, checking via XGBoost...")
        
        # Alternative check via XGBoost
        try:
            test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            test_data = xgb.DMatrix(np.random.randn(100, 10), label=np.random.randn(100))
            xgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
            print("✅ GPU is available and will be used for training")
            return True
        except Exception as e:
            print(f"⚠️ GPU not available for XGBoost: {e}")
            print("   Will use CPU for training")
            return False
    
    def prepare_data(self):
        """Load and prepare 30-minute intraday data for XGBoost"""
        print("\nLoading 30-minute intraday volatility data for XGBoost...")
        
        # Use the same data files as other models
        vol_file = 'processed_data/vols_mats_30min_standardized.h5'
        volvol_file = 'processed_data/volvols_mats_30min_standardized.h5'
        
        # Check if files exist
        if not os.path.exists(vol_file):
            raise FileNotFoundError(f"30-minute data not found: {vol_file}\nPlease run scripts 2 and 4 first!")
        
        # Create datasets
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
        
        # Convert to XGBoost format
        self._create_xgboost_datasets()
    
    def _create_xgboost_datasets(self):
        """Convert PyTorch datasets to XGBoost DMatrix format with feature engineering"""
        print("\nPreparing data for XGBoost (feature engineering)...")
        
        # Process training data
        X_train, y_train = self._extract_features(self.train_dataset, "Training")
        self.dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        # Process validation data
        X_val, y_val = self._extract_features(self.val_dataset, "Validation")
        self.dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        
        # Process test data
        X_test, y_test = self._extract_features(self.test_dataset, "Test")
        self.dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_names)
        
        print(f"\nFeature dimensions: {X_train.shape[1]} features")
        print(f"Feature types:")
        print(f"  - Raw features: {self.config['seq_length'] * 930}")
        print(f"  - Engineered features: {len(self.feature_names) - self.config['seq_length'] * 930}")
    
    def _extract_features(self, dataset, split_name):
        """
        Extract and engineer features for XGBoost
        
        Additional features for tree-based models:
        1. Statistical aggregations across time
        2. Volatility momentum and trends
        3. Cross-sectional features
        """
        all_features = []
        all_targets = []
        
        for i in tqdm(range(len(dataset)), desc=f"Processing {split_name} samples"):
            sample = dataset[i]
            features = sample['features'].numpy()  # Shape: [42, 930]
            target = sample['target'].numpy()      # Shape: [30]
            
            # Flatten time series features
            flat_features = features.flatten()
            
            # Engineer additional features
            engineered = self._engineer_features(features)
            
            # Combine all features
            combined_features = np.concatenate([flat_features, engineered])
            
            # For multi-output, we'll train 30 models (one per stock)
            # For now, use mean volatility as single target
            mean_target = np.mean(target)
            
            all_features.append(combined_features)
            all_targets.append(mean_target)
        
        X = np.vstack(all_features)
        y = np.array(all_targets)
        
        return X, y
    
    def _engineer_features(self, features):
        """
        Engineer additional features for XGBoost
        
        Args:
            features: [seq_length, 930] array
            
        Returns:
            Engineered features array
        """
        engineered = []
        
        # Split features into components
        seq_length = features.shape[0]
        vols = features[:, :30]  # Volatilities
        covols = features[:, 30:465]  # Covolatilities
        volvols = features[:, 465:495]  # Vol-of-vols
        covolvols = features[:, 495:]  # Covol-of-vols
        
        # 1. Volatility statistics across time
        engineered.extend([
            vols.mean(axis=0).flatten(),  # Mean vol per stock
            vols.std(axis=0).flatten(),   # Std vol per stock
            vols[-1, :].flatten(),         # Last vol
            vols[-6:, :].mean(axis=0).flatten(),  # Recent mean (6 intervals = 3 hours)
        ])
        
        # 2. Volatility trends
        vol_diff = np.diff(vols, axis=0)
        engineered.extend([
            vol_diff.mean(axis=0).flatten(),  # Average change
            vol_diff[-1, :].flatten(),        # Last change
        ])
        
        # 3. Vol-of-vol features
        engineered.extend([
            volvols.mean(axis=0).flatten(),
            volvols[-1, :].flatten(),
        ])
        
        # 4. Cross-sectional features
        engineered.extend([
            np.mean(vols, axis=1),  # Market vol at each time
            np.std(vols, axis=1),   # Cross-sectional dispersion
        ])
        
        # 5. Covariance features (simplified)
        engineered.extend([
            covols.mean(axis=0)[:100],  # Sample of mean covariances
            covols.std(axis=0)[:100],   # Sample of cov volatility
        ])
        
        engineered_array = np.concatenate(engineered)
        
        # Create feature names on first call
        if not hasattr(self, 'feature_names'):
            self._create_feature_names(seq_length, engineered_array.shape[0])
        
        return engineered_array
    
    def _create_feature_names(self, seq_length, n_engineered):
        """Create feature names for interpretability"""
        self.feature_names = []
        
        # Raw features
        for t in range(seq_length):
            for i in range(930):
                self.feature_names.append(f'raw_t{t}_f{i}')
        
        # Engineered features
        eng_names = [
            *[f'vol_mean_s{i}' for i in range(30)],
            *[f'vol_std_s{i}' for i in range(30)],
            *[f'vol_last_s{i}' for i in range(30)],
            *[f'vol_recent_s{i}' for i in range(30)],
            *[f'vol_diff_mean_s{i}' for i in range(30)],
            *[f'vol_diff_last_s{i}' for i in range(30)],
            *[f'volvol_mean_s{i}' for i in range(30)],
            *[f'volvol_last_s{i}' for i in range(30)],
        ]
        
        # Add remaining engineered features
        remaining = n_engineered - len(eng_names)
        eng_names.extend([f'eng_{i}' for i in range(remaining)])
        
        self.feature_names.extend(eng_names[:n_engineered])
    
    def tune_hyperparameters(self, n_iter=20):
        """
        Tune XGBoost hyperparameters using RandomizedSearchCV
        
        Args:
            n_iter: Number of parameter combinations to try
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Define parameter distributions
        param_distributions = {
            'max_depth': [4, 6, 8, 10, 12],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
            'n_estimators': [300, 500, 700, 1000],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.01, 0.1, 0.5],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0],
        }
        
        # Create base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='gpu_hist' if self.use_gpu else 'hist',
            predictor='gpu_predictor' if self.use_gpu else 'cpu_predictor',
            random_state=self.seed,
            n_jobs=-1 if not self.use_gpu else 1,
            gpu_id=0 if self.use_gpu else None,
        )
        
        # Get training data
        X_train = self.dtrain.get_data().toarray() if hasattr(self.dtrain.get_data(), 'toarray') else self.dtrain.get_data()
        y_train = self.dtrain.get_label()
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=1,
            random_state=self.seed,
            n_jobs=1  # Use 1 job since XGBoost handles parallelization
        )
        
        print(f"Testing {n_iter} parameter combinations with 3-fold CV...")
        random_search.fit(X_train, y_train)
        
        # Update parameters with best found
        self.base_params.update(random_search.best_params_)
        
        print(f"\nBest parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest CV score: {-random_search.best_score_:.6f}")
        
        # Save tuning results
        tuning_results = {
            'best_params': random_search.best_params_,
            'best_score': float(-random_search.best_score_),
            'n_iter': n_iter,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'tuning_results.json'), 'w') as f:
            json.dump(tuning_results, f, indent=2)
    
    def train(self, tune_hyperparams=False):
        """
        Train XGBoost model with early stopping
        
        Args:
            tune_hyperparams: Whether to tune hyperparameters first
        """
        print("\n" + "="*80)
        print("XGBOOST TRAINING - 30-MINUTE INTRADAY VOLATILITY")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"GPU: {'✅ Enabled' if self.use_gpu else '❌ Disabled (using CPU)'}")
        print("="*80)
        
        # Optionally tune hyperparameters
        if tune_hyperparams:
            self.tune_hyperparameters(n_iter=20)
        
        # Setup evaluation list
        evals = [(self.dtrain, 'train'), (self.dval, 'validation')]
        
        # Training with early stopping
        print("\nTraining XGBoost model with early stopping...")
        
        self.model = xgb.train(
            self.base_params,
            self.dtrain,
            num_boost_round=self.base_params.get('n_estimators', 500),
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=10,
            callbacks=[self._custom_callback()]
        )
        
        # Save model
        model_path = os.path.join(self.output_dir, 'xgboost_model.json')
        self.model.save_model(model_path)
        print(f"\n✅ Model saved to: {model_path}")
        
        # Also save in joblib format for sklearn compatibility
        joblib_path = os.path.join(self.output_dir, 'xgboost_model.pkl')
        joblib.dump(self.model, joblib_path)
        
        # Plot training curves
        self._plot_training_curves()
        
        # Feature importance analysis
        self._analyze_feature_importance()
        
        return self.model
    
    def _custom_callback(self):
        """Custom callback to track QLIKE during training"""
        def callback(env):
            # Get current iteration results
            for data, metric in env.evaluation_result_list:
                if 'validation' in data:
                    # Calculate QLIKE on validation set
                    y_pred = self.model.predict(self.dval)
                    y_true = self.dval.get_label()
                    
                    # Since we're predicting mean, expand to 30 stocks
                    y_pred_expanded = np.tile(y_pred.reshape(-1, 1), (1, 30))
                    y_true_expanded = np.tile(y_true.reshape(-1, 1), (1, 30))
                    
                    # Calculate QLIKE
                    metrics = self.evaluator.calculate_all_metrics(
                        y_pred_expanded, y_true_expanded, is_variance=True
                    )
                    
                    self.val_history['qlike'].append(metrics['qlike'])
                    
                    if env.iteration % 10 == 0:
                        print(f"  [Iteration {env.iteration}] Val QLIKE: {metrics['qlike']:.4f}")
        
        return callback
    
    def _plot_training_curves(self):
        """Plot and save training curves"""
        results = self.model.evals_result()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE curves
        ax1.plot(results['train']['rmse'], label='Train RMSE', alpha=0.8)
        ax1.plot(results['validation']['rmse'], label='Val RMSE', alpha=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Training & Validation RMSE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Feature importance
        importance = self.model.get_score(importance_type='gain')
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        
        ax2.barh(range(len(top_features)), [f[1] for f in top_features])
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels([f[0] for f in top_features])
        ax2.set_xlabel('Gain')
        ax2.set_title('Top 20 Feature Importance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=100)
        plt.close()
        
        print(f"✅ Training curves saved to: {self.output_dir}/training_curves.png")
    
    def _analyze_feature_importance(self):
        """Analyze and save feature importance"""
        # Get feature importance
        importance_gain = self.model.get_score(importance_type='gain')
        importance_weight = self.model.get_score(importance_type='weight')
        importance_cover = self.model.get_score(importance_type='cover')
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_gain.keys()),
            'gain': list(importance_gain.values()),
            'weight': [importance_weight.get(f, 0) for f in importance_gain.keys()],
            'cover': [importance_cover.get(f, 0) for f in importance_gain.keys()]
        })
        
        importance_df = importance_df.sort_values('gain', ascending=False)
        
        # Save to CSV
        importance_path = os.path.join(self.output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        
        print(f"\nTop 10 Most Important Features (by gain):")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['gain']:10.2f}")
    
    def evaluate_test(self):
        """Evaluate model on test set with comprehensive metrics"""
        print("\n" + "="*60)
        print("TEST SET EVALUATION - XGBOOST")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(self.dtest)
        y_true = self.dtest.get_label()
        
        # Expand predictions to 30 stocks (since we predicted mean)
        y_pred_expanded = np.tile(y_pred.reshape(-1, 1), (1, 30))
        y_true_expanded = np.tile(y_true.reshape(-1, 1), (1, 30))
        
        # Calculate metrics
        metrics = self.evaluator.calculate_all_metrics(
            y_pred_expanded, y_true_expanded, is_variance=True
        )
        
        # Calculate R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        print(f"Test RMSE (standardized): {np.sqrt(mean_squared_error(y_true, y_pred)):.6f}")
        print(f"Test RMSE (volatility): {metrics['rmse_vol']:.6f}")
        print(f"Test MAE (volatility): {metrics['mae_vol']:.6f}")
        print(f"Test QLIKE: {metrics['qlike']:.4f}")
        print(f"Test R² Score: {r2_score:.4f}")
        print(f"Test samples: {len(self.test_dataset)}")
        print("="*60)
        
        # Save test results
        test_results = {
            'test_rmse_std': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'test_rmse_vol': float(metrics['rmse_vol']),
            'test_mae_vol': float(metrics['mae_vol']),
            'test_qlike': float(metrics['qlike']),
            'test_r2': float(r2_score),
            'test_samples': len(self.test_dataset),
            'model_params': self.base_params,
            'best_iteration': self.model.best_iteration,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n✅ Test results saved to: {self.output_dir}/test_results.json")
        
        return test_results


def main():
    """Main execution"""
    print("="*80)
    print("XGBOOST TRAINING FOR 30-MINUTE INTRADAY VOLATILITY")
    print("="*80)
    
    # Initialize trainer
    trainer = XGBoostVolatilityTrainer()
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model (with optional hyperparameter tuning)
    trainer.train(tune_hyperparams=True)
    
    # Evaluate on test set
    trainer.evaluate_test()
    
    print("\n" + "="*80)
    print("✅ XGBOOST TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()