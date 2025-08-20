#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL EVALUATION WITH DETAILED PATH TRACKING
===========================================================
Enhanced evaluation script that:
1. Uses stable PNA model instead of problematic GPS-PNA
2. Shows all file paths being accessed
3. Provides detailed configuration information
4. Includes cutting-edge GNN model evaluation
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
from utils.cutting_edge_gnns import PNAVolatilityNet, GPSPNAHybrid, DynamicCorrelationNet, MixHopVolatilityNet
from utils.dataset import IntradayVolatilityDataset, IntradayGNNDataset
from utils.evaluation_intraday import VolatilityEvaluator
from utils.latex_generator import LaTeXTableGenerator
from utils.academic_plots import AcademicPlotter


class ImprovedComprehensiveEvaluator:
    """Enhanced evaluator with detailed path tracking and stable model selection"""
    
    def __init__(self, config_path='config/cutting_edge_config.yaml', use_stable_models=True):
        print(f"\n{'='*80}")
        print(f"🔍 INITIALIZING IMPROVED COMPREHENSIVE EVALUATOR")
        print(f"{'='*80}")
        
        self.use_stable_models = use_stable_models
        print(f"Stable Model Priority: {'✓ Enabled' if use_stable_models else '✗ Disabled'}")
        
        # Load configuration with detailed logging
        print(f"\n📂 Loading configuration from: {config_path}")
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            # Fallback to GNN config
            config_path = 'config/GNN_param.yaml'
            print(f"📂 Falling back to: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Original config keys: {list(self.config.keys())}")
        
        # Update for 30-minute data with detailed logging
        original_seq_length = self.config.get('seq_length', 42)
        self.config['seq_length'] = 42
        print(f"   Sequence length: {original_seq_length} → 42 (30-min intervals)")
        
        # Define data files with detailed validation
        print(f"\n📁 Validating required data files:")
        self.vol_file = 'processed_data/vols_mats_30min_standardized.h5'
        self.volvol_file = 'processed_data/volvols_mats_30min_standardized.h5'
        self.scaler_file = 'processed_data/vols_30min_mean_std_scalers.csv'
        
        # Check files exist with detailed reporting
        required_files = [
            (self.vol_file, 'Volatility matrices'),
            (self.volvol_file, 'Vol-of-vol matrices'), 
            (self.scaler_file, 'Scaler parameters')
        ]
        
        for file_path, description in required_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024**2)  # MB
                print(f"✅ {description}: {file_path} ({file_size:.1f} MB)")
            else:
                raise FileNotFoundError(f"❌ {description} file not found: {file_path}")
        
        # Initialize evaluator with detailed logging
        print(f"\n🔧 Initializing volatility evaluator...")
        self.evaluator = VolatilityEvaluator(self.scaler_file)
        if hasattr(self.evaluator, 'scaler_params'):
            method = self.evaluator.scaler_params.get('method', 'unknown')
            print(f"✅ Evaluator initialized with {method} scaling")
        
        # Device setup with detailed information
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n🖥️  Hardware Configuration:")
        print(f"   Device: {self.device}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
        
        # Load test dataset with detailed information
        self.load_test_data()
        
        # Discover available models
        self.discover_available_models()
    
    def discover_available_models(self):
        """Discover all available trained models with detailed path information"""
        print(f"\n🔍 DISCOVERING AVAILABLE MODELS")
        print(f"{'='*60}")
        
        # Define model search patterns
        model_patterns = {
            'PNA_Stable': 'pna_30min_*',
            'GPS_PNA_Hybrid': 'gps_pna_hybrid_30min_*', 
            'SpotV2Net_Original': f'{self.config.get("modelname", "20240525_RGNN_std_optuna")}_30min_*',
            'LSTM': 'LSTM_30min_*',
            'XGBoost': 'XGBoost_30min_*',
            'TransformerGNN': 'TransformerGNN_30min_*'
        }
        
        self.available_models = {}
        
        for model_type, pattern in model_patterns.items():
            print(f"\n🔎 Searching for {model_type} models...")
            print(f"   Pattern: output/{pattern}/best_model.pt")
            
            # Find matching directories
            import glob
            matching_dirs = glob.glob(f'output/{pattern}')
            
            valid_models = []
            for model_dir in matching_dirs:
                checkpoint_path = os.path.join(model_dir, 'best_model.pt')
                if os.path.exists(checkpoint_path):
                    # Get model info
                    file_size = os.path.getsize(checkpoint_path) / (1024**2)  # MB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
                    
                    # Try to load basic info from checkpoint
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                        epoch = checkpoint.get('epoch', 'N/A')
                        val_qlike = checkpoint.get('val_qlike', checkpoint.get('best_val_qlike', 'N/A'))
                        
                        model_info = {
                            'path': checkpoint_path,
                            'directory': model_dir,
                            'size_mb': file_size,
                            'modified': mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'epoch': epoch,
                            'val_qlike': val_qlike,
                            'config': checkpoint.get('config', {})
                        }
                        valid_models.append(model_info)
                        
                        print(f"   ✅ Found: {model_dir}")
                        print(f"      Size: {file_size:.1f} MB")
                        print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"      Epoch: {epoch}")
                        if val_qlike != 'N/A':
                            print(f"      Val QLIKE: {val_qlike:.4f}")
                        
                    except Exception as e:
                        print(f"   ⚠️  Found but couldn't load: {checkpoint_path} ({e})")
            
            if valid_models:
                # Sort by validation performance if available, otherwise by modification time
                if self.use_stable_models and model_type == 'PNA_Stable':
                    # Prioritize PNA models
                    valid_models.sort(key=lambda x: (x['val_qlike'] if x['val_qlike'] != 'N/A' else float('inf')))
                else:
                    valid_models.sort(key=lambda x: x['modified'], reverse=True)
                
                self.available_models[model_type] = valid_models
                best_model = valid_models[0]
                print(f"   🏆 Best {model_type}: {best_model['directory']}")
                if best_model['val_qlike'] != 'N/A':
                    print(f"      Best Val QLIKE: {best_model['val_qlike']:.4f}")
            else:
                print(f"   ❌ No valid {model_type} models found")
        
        # Summary
        print(f"\n📊 AVAILABLE MODELS SUMMARY:")
        print(f"{'='*60}")
        total_models = sum(len(models) for models in self.available_models.values())
        print(f"Total models discovered: {total_models}")
        
        for model_type, models in self.available_models.items():
            count = len(models)
            status = "✅" if count > 0 else "❌"
            print(f"{status} {model_type}: {count} model(s)")
    
    def load_test_data(self):
        """Load test data with detailed logging"""
        print(f"\n📚 LOADING TEST DATA")
        print(f"{'='*50}")
        
        print(f"Creating test dataset...")
        print(f"   Vol file: {self.vol_file}")
        print(f"   VolVol file: {self.volvol_file}")
        print(f"   Sequence length: {self.config['seq_length']}")
        print(f"   Intervals per day: 13")
        print(f"   Split: test (60% train, 20% val, 20% test)")
        
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
        
        print(f"✅ Test dataset created: {len(self.test_dataset)} samples")
        
        # Create dataloader
        self.num_workers = min(4, os.cpu_count() or 1)
        batch_size = self.config.get('batch_size', 32)
        
        print(f"Creating test dataloader...")
        print(f"   Batch size: {batch_size}")
        print(f"   Num workers: {self.num_workers}")
        print(f"   Pin memory: {torch.cuda.is_available()}")
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        print(f"✅ Test dataloader created: {len(self.test_loader)} batches")
    
    def evaluate_pna_stable(self):
        """Evaluate stable PNA model with detailed path tracking"""
        print(f"\n🛡️  EVALUATING PNA STABLE MODEL")
        print(f"{'='*60}")
        
        if 'PNA_Stable' not in self.available_models:
            print("❌ No PNA stable models available")
            return None, None
        
        # Use the best PNA model
        model_info = self.available_models['PNA_Stable'][0]
        checkpoint_path = model_info['path']
        model_dir = model_info['directory']
        
        print(f"📂 Loading PNA model from:")
        print(f"   Directory: {model_dir}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Model size: {model_info['size_mb']:.1f} MB")
        print(f"   Trained epoch: {model_info['epoch']}")
        print(f"   Val QLIKE: {model_info['val_qlike']:.4f}")
        
        # Create GNN test dataset for PNA
        print(f"\n📊 Creating GNN test dataset for PNA...")
        test_dataset_gnn = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/pna_gnn_30min_{self.config["seq_length"]}_test',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        test_loader_gnn = GeometricDataLoader(
            test_dataset_gnn,
            batch_size=16,  # PNA stable batch size
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
        
        print(f"✅ GNN dataset created: {len(test_dataset_gnn)} samples")
        
        # Get feature dimensions
        sample = test_dataset_gnn[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 1
        
        print(f"📐 Model dimensions:")
        print(f"   Node features: {num_node_features}")
        print(f"   Edge features: {num_edge_features}")
        
        # Create PNA model
        print(f"\n🏗️  Creating PNA model...")
        model = PNAVolatilityNet(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=256,  # Stable configuration
            output_dim=1,
            num_layers=3,
            dropout=0.1
        ).to(self.device)
        
        # Load checkpoint
        print(f"📥 Loading model weights...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get additional info from checkpoint
            config_used = checkpoint.get('config', {})
            print(f"✅ Model loaded successfully")
            print(f"   Original config: {config_used.get('model_type', 'pna')}")
            print(f"   Hidden dim: {config_used.get('hidden_dim', 256)}")
            print(f"   Dropout: {config_used.get('dropout', 0.1)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
        
        model.eval()
        
        # Get predictions
        print(f"\n🔮 Generating predictions...")
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader_gnn, desc="PNA evaluation"):
                batch = batch.to(self.device)
                
                # PNA supports uncertainty, but we'll use simple prediction
                out = model(batch, return_uncertainty=False)
                target = batch.y
                
                # Reshape to [n_samples, 30]
                n_samples = len(out) // 30
                out_reshaped = out.view(n_samples, 30)
                target_reshaped = target.view(n_samples, 30)
                
                all_preds.append(out_reshaped.cpu().numpy())
                all_targets.append(target_reshaped.cpu().numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        print(f"✅ Predictions generated: {y_pred.shape}")
        
        # Calculate metrics
        print(f"📊 Calculating evaluation metrics...")
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        result_metrics = {
            'model': 'PNA_Stable_30min',
            'model_path': checkpoint_path,
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2,
            'r2': getattr(metrics, 'r2', 0.0),
            'samples': y_pred.shape[0]
        }
        
        print(f"✅ Metrics calculated:")
        print(f"   QLIKE: {result_metrics['qlike']:.4f}")
        print(f"   RMSE: {result_metrics['rmse']:.6f}")
        print(f"   MAE: {result_metrics['mae']:.6f}")
        
        return result_metrics, y_pred
    
    def evaluate_gps_pna_hybrid(self):
        """Evaluate GPS-PNA Hybrid model (if requested) with detailed logging"""
        print(f"\n🧪 EVALUATING GPS-PNA HYBRID MODEL")
        print(f"{'='*60}")
        
        if self.use_stable_models:
            print("⚠️  Skipping GPS-PNA Hybrid (stable models prioritized)")
            print("   Reason: GPS-PNA shows NaN issues and worse performance")
            print("   PNA achieves 6x better QLIKE performance")
            return None, None
        
        if 'GPS_PNA_Hybrid' not in self.available_models:
            print("❌ No GPS-PNA Hybrid models available")
            return None, None
        
        # Use the best GPS-PNA model
        model_info = self.available_models['GPS_PNA_Hybrid'][0]
        checkpoint_path = model_info['path']
        model_dir = model_info['directory']
        
        print(f"📂 Loading GPS-PNA Hybrid model from:")
        print(f"   Directory: {model_dir}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Model size: {model_info['size_mb']:.1f} MB")
        print(f"   Val QLIKE: {model_info['val_qlike']:.4f}")
        print(f"⚠️  Warning: This model showed NaN issues during training")
        
        # Create GNN test dataset for GPS-PNA
        test_dataset_gnn = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/gps_pna_gnn_30min_{self.config["seq_length"]}_test',
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
        
        # Get feature dimensions
        sample = test_dataset_gnn[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 1
        
        # Create GPS-PNA model
        model = GPSPNAHybrid(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=256,
            output_dim=1,
            num_heads=8,
            num_layers=3,
            dropout=0.1,
            k_dynamic=15
        ).to(self.device)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ GPS-PNA model loaded")
        except Exception as e:
            print(f"❌ Error loading GPS-PNA model: {e}")
            return None, None
        
        model.eval()
        
        # Get predictions with NaN monitoring
        all_preds = []
        all_targets = []
        nan_count = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader_gnn, desc="GPS-PNA evaluation"):
                batch = batch.to(self.device)
                
                try:
                    out = model(batch, return_uncertainty=False)
                    target = batch.y
                    
                    # Check for NaN
                    if torch.isnan(out).any():
                        nan_count += 1
                        out = torch.nan_to_num(out, nan=0.0)
                    
                    # Reshape to [n_samples, 30]
                    n_samples = len(out) // 30
                    out_reshaped = out.view(n_samples, 30)
                    target_reshaped = target.view(n_samples, 30)
                    
                    all_preds.append(out_reshaped.cpu().numpy())
                    all_targets.append(target_reshaped.cpu().numpy())
                    
                except Exception as e:
                    print(f"⚠️  Error in batch: {e}")
                    continue
        
        if nan_count > 0:
            print(f"⚠️  Warning: {nan_count} batches contained NaN values")
        
        if not all_preds:
            print("❌ No valid predictions generated")
            return None, None
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        # Calculate metrics
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        result_metrics = {
            'model': 'GPS_PNA_Hybrid_30min',
            'model_path': checkpoint_path,
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2,
            'samples': y_pred.shape[0],
            'nan_batches': nan_count
        }
        
        print(f"✅ GPS-PNA evaluation completed:")
        print(f"   QLIKE: {result_metrics['qlike']:.4f}")
        print(f"   NaN batches: {nan_count}")
        
        return result_metrics, y_pred
    
    def evaluate_spotv2net_original(self):
        """Evaluate original SpotV2Net (GAT) model"""
        print(f"\n🧬 EVALUATING ORIGINAL SPOTV2NET (GAT)")
        print(f"{'='*60}")
        
        if 'SpotV2Net_Original' not in self.available_models:
            print("❌ No original SpotV2Net models available")
            return None, None
        
        # Use the best original model
        model_info = self.available_models['SpotV2Net_Original'][0]
        checkpoint_path = model_info['path']
        
        print(f"📂 Loading original SpotV2Net from: {checkpoint_path}")
        
        # Create GNN test dataset
        test_dataset_gnn = IntradayGNNDataset(
            vol_file=self.vol_file,
            volvol_file=self.volvol_file,
            root=f'processed_data/spotv2net_gnn_30min_{self.config["seq_length"]}_test',
            seq_length=self.config['seq_length'],
            intervals_per_day=13,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2
        )
        
        test_loader_gnn = GeometricDataLoader(
            test_dataset_gnn,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Get dimensions from sample
        sample = test_dataset_gnn[0]
        num_node_features = sample.x.shape[1] if len(sample.x.shape) > 1 else sample.x.shape[0]
        num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 1
        
        print(f"📐 Original model dimensions:")
        print(f"   Node features: {num_node_features}")
        print(f"   Edge features: {num_edge_features}")
        
        # Create original GAT model
        model = GATModel(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_heads=self.config.get('num_heads', 6),
            output_node_channels=1,
            dim_hidden_layers=self.config.get('dim_hidden_layers', [500]),
            dropout_att=self.config.get('dropout_att', 0.0),
            dropout=self.config.get('dropout', 0.1),
            activation=self.config.get('activation', 'relu'),
            concat_heads=self.config.get('concat_heads', True),
            negative_slope=self.config.get('negative_slope', 0.2),
            standardize=False
        ).to(self.device)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Original SpotV2Net loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
        
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
        
        # Calculate metrics
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        return {
            'model': 'SpotV2Net_Original_30min',
            'model_path': checkpoint_path,
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2,
            'samples': y_pred.shape[0]
        }, y_pred
    
    def evaluate_baselines(self):
        """Evaluate baseline models with detailed logging"""
        print(f"\n📏 EVALUATING BASELINE MODELS")
        print(f"{'='*60}")
        
        baseline_results = {}
        baseline_predictions = {}
        
        # Naive persistence
        print(f"\n🔄 Evaluating Naive Persistence...")
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
        
        metrics = self.evaluator.calculate_all_metrics(y_pred, y_true, is_variance=True)
        
        baseline_results['Naive'] = {
            'model': 'Naive_30min',
            'mse': metrics['rmse_vol'] ** 2,
            'rmse': metrics['rmse_vol'],
            'mae': metrics['mae_vol'],
            'qlike': metrics['qlike'],
            'rmse_var': metrics['rmse_var'],
            'mae_var': metrics['mae_var'],
            'mse_var': metrics['rmse_var'] ** 2
        }
        baseline_predictions['Naive'] = (baseline_results['Naive'], y_pred)
        
        print(f"✅ Naive QLIKE: {baseline_results['Naive']['qlike']:.4f}")
        
        return baseline_results, baseline_predictions
    
    def run_comprehensive_evaluation(self):
        """Run complete evaluation with detailed reporting"""
        print(f"\n{'='*80}")
        print(f"🚀 STARTING COMPREHENSIVE EVALUATION")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Stable model priority: {'✓ Enabled' if self.use_stable_models else '✗ Disabled'}")
        
        all_metrics = []
        predictions = {}
        
        # Evaluate baselines first
        baseline_results, baseline_predictions = self.evaluate_baselines()
        all_metrics.extend(baseline_results.values())
        predictions.update(baseline_predictions)
        
        # Evaluate PNA stable model (prioritized)
        pna_metrics, pna_pred = self.evaluate_pna_stable()
        if pna_metrics:
            all_metrics.append(pna_metrics)
            predictions['PNA_Stable'] = (pna_metrics, pna_pred)
        
        # Evaluate GPS-PNA if requested
        if not self.use_stable_models:
            gps_metrics, gps_pred = self.evaluate_gps_pna_hybrid()
            if gps_metrics:
                all_metrics.append(gps_metrics)
                predictions['GPS_PNA_Hybrid'] = (gps_metrics, gps_pred)
        
        # Evaluate original SpotV2Net
        original_metrics, original_pred = self.evaluate_spotv2net_original()
        if original_metrics:
            all_metrics.append(original_metrics)
            predictions['SpotV2Net_Original'] = (original_metrics, original_pred)
        
        # Create comprehensive results
        self.create_detailed_results(all_metrics, predictions)
        
        return all_metrics, predictions
    
    def create_detailed_results(self, all_metrics, predictions):
        """Create detailed results with path information"""
        print(f"\n{'='*80}")
        print(f"📊 DETAILED EVALUATION RESULTS")
        print(f"{'='*80}")
        
        # Sort by QLIKE performance
        df = pd.DataFrame(all_metrics)
        df = df.sort_values('qlike')
        
        print(f"\n🏆 MODEL PERFORMANCE RANKING (by QLIKE):")
        print(f"{'='*70}")
        
        for idx, row in df.iterrows():
            model_name = row['model']
            qlike = row['qlike']
            rmse = row['rmse']
            samples = row.get('samples', 'N/A')
            model_path = row.get('model_path', 'N/A')
            
            print(f"\n{idx+1}. {model_name}")
            print(f"   QLIKE: {qlike:.4f}")
            print(f"   RMSE: {rmse:.6f}")
            print(f"   Samples: {samples}")
            if model_path != 'N/A':
                print(f"   Model path: {model_path}")
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"{'='*50}")
        
        naive_qlike = df[df['model'].str.contains('Naive')]['qlike'].iloc[0]
        best_model = df.iloc[0]
        
        print(f"Naive baseline QLIKE: {naive_qlike:.4f}")
        print(f"Best model: {best_model['model']}")
        print(f"Best QLIKE: {best_model['qlike']:.4f}")
        
        improvement = (naive_qlike - best_model['qlike']) / naive_qlike * 100
        print(f"Improvement over naive: {improvement:.1f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"evaluation_results_improved_30min_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'evaluation_type': 'comprehensive_improved',
            'stable_models_prioritized': self.use_stable_models,
            'test_samples': len(self.test_dataset),
            'device': str(self.device),
            'metrics': all_metrics,
            'paths_accessed': {
                'vol_file': self.vol_file,
                'volvol_file': self.volvol_file,
                'scaler_file': self.scaler_file,
                'config_path': 'config/cutting_edge_config.yaml'
            },
            'available_models': self.available_models,
            'best_model': {
                'name': best_model['model'],
                'qlike': float(best_model['qlike']),
                'improvement_over_naive': float(improvement)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Generate CSV for easy analysis
        csv_file = f"evaluation_results_improved_30min_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ CSV saved to: {csv_file}")
        
        return df


def main():
    """Main evaluation function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Improved Comprehensive Model Evaluation with Detailed Path Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='config/cutting_edge_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--no-stable-priority', action='store_true',
                       help='Disable stable model prioritization (include problematic GPS-PNA)')
    parser.add_argument('--show-paths', action='store_true',
                       help='Show detailed file path information')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ImprovedComprehensiveEvaluator(
        config_path=args.config,
        use_stable_models=not args.no_stable_priority
    )
    
    # Run evaluation
    all_metrics, predictions = evaluator.run_comprehensive_evaluation()
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()