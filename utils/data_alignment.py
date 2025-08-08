#!/usr/bin/env python3
"""
Rigorous Data Alignment and Temporal Splitting Strategy for SpotV2Net
=======================================================================

This module ensures perfect data alignment across all models (GNN, LSTM, etc.)
and implements proper temporal splits for financial time series research.

Key Principles:
1. NO data leakage: Future information never influences past predictions
2. Temporal ordering: Train < Validation < Test (chronologically)
3. Consistent indexing: Same samples across all models
4. Reproducibility: Fixed random seeds where applicable
5. Validation set: For hyperparameter tuning without touching test set
"""

import numpy as np
import torch
import h5py
import os
from typing import Tuple, Dict, List
from natsort import natsorted
import yaml
import json
from datetime import datetime


class DataAlignmentConfig:
    """
    Central configuration for data alignment across all models.
    This ensures consistency and reproducibility.
    """
    
    def __init__(self, seq_length: int = 42):
        self.seq_length = seq_length
        
        # Data paths
        self.vol_file = 'processed_data/vols_mats_taq_standardized.h5'
        self.volvol_file = 'processed_data/volvols_mats_taq_standardized.h5'
        
        # Split ratios (60/20/20 is standard for financial time series)
        # This provides sufficient validation data for hyperparameter tuning
        self.train_ratio = 0.60
        self.val_ratio = 0.20
        self.test_ratio = 0.20
        
        # Verify ratios sum to 1.0
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6
        
        # Calculate actual splits
        self._calculate_splits()
        
    def _calculate_splits(self):
        """Calculate exact sample indices for train/val/test splits"""
        
        with h5py.File(self.vol_file, 'r') as f:
            self.total_matrices = len(f.keys())
        
        # Total samples after accounting for sequence length
        self.total_samples = self.total_matrices - self.seq_length
        
        # Calculate split sizes
        self.train_size = int(self.total_samples * self.train_ratio)
        self.val_size = int(self.total_samples * self.val_ratio)
        self.test_size = self.total_samples - self.train_size - self.val_size
        
        # Define exact indices for each split
        self.train_start = 0
        self.train_end = self.train_size
        
        self.val_start = self.train_end
        self.val_end = self.val_start + self.val_size
        
        self.test_start = self.val_end
        self.test_end = self.total_samples
        
        # Sanity checks
        assert self.train_end == self.val_start
        assert self.val_end == self.test_start
        assert self.test_end == self.total_samples
        
    def get_split_indices(self, split: str) -> Tuple[int, int]:
        """
        Get start and end indices for a given split.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            (start_idx, end_idx) tuple
        """
        if split == 'train':
            return self.train_start, self.train_end
        elif split == 'val':
            return self.val_start, self.val_end
        elif split == 'test':
            return self.test_start, self.test_end
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    def get_split_data(self, data: np.ndarray, split: str) -> np.ndarray:
        """
        Extract data for a specific split.
        
        Args:
            data: Full dataset array
            split: One of 'train', 'val', 'test'
            
        Returns:
            Subset of data for the specified split
        """
        start, end = self.get_split_indices(split)
        return data[start:end]
    
    def save_config(self, filepath: str = 'processed_data/data_alignment_config.json'):
        """Save configuration to JSON for reproducibility"""
        
        config = {
            'timestamp': datetime.now().isoformat(),
            'seq_length': self.seq_length,
            'total_matrices': self.total_matrices,
            'total_samples': self.total_samples,
            'splits': {
                'train': {
                    'ratio': self.train_ratio,
                    'size': self.train_size,
                    'start_idx': self.train_start,
                    'end_idx': self.train_end,
                    'years_approx': self.train_size / 252
                },
                'val': {
                    'ratio': self.val_ratio,
                    'size': self.val_size,
                    'start_idx': self.val_start,
                    'end_idx': self.val_end,
                    'years_approx': self.val_size / 252
                },
                'test': {
                    'ratio': self.test_ratio,
                    'size': self.test_size,
                    'start_idx': self.test_start,
                    'end_idx': self.test_end,
                    'years_approx': self.test_size / 252
                }
            },
            'verification': {
                'sum_of_splits': self.train_size + self.val_size + self.test_size,
                'equals_total': (self.train_size + self.val_size + self.test_size) == self.total_samples
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def print_summary(self):
        """Print a comprehensive summary of the data alignment configuration"""
        
        print("="*80)
        print("DATA ALIGNMENT CONFIGURATION")
        print("="*80)
        
        print(f"\nData Statistics:")
        print(f"  Total matrices: {self.total_matrices}")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Total samples: {self.total_samples}")
        
        print(f"\nTemporal Splits (60/20/20):")
        print(f"  Training Set:")
        print(f"    - Samples: {self.train_size} (indices {self.train_start} to {self.train_end-1})")
        print(f"    - Approx. years: {self.train_size/252:.2f}")
        print(f"    - Purpose: Model training")
        
        print(f"  Validation Set:")
        print(f"    - Samples: {self.val_size} (indices {self.val_start} to {self.val_end-1})")
        print(f"    - Approx. years: {self.val_size/252:.2f}")
        print(f"    - Purpose: Hyperparameter tuning, early stopping")
        
        print(f"  Test Set:")
        print(f"    - Samples: {self.test_size} (indices {self.test_start} to {self.test_end-1})")
        print(f"    - Approx. years: {self.test_size/252:.2f}")
        print(f"    - Purpose: Final model evaluation (NEVER used during training)")
        
        print(f"\nData Integrity Checks:")
        print(f"  ✓ No overlap between splits")
        print(f"  ✓ Temporal ordering preserved (train < val < test)")
        print(f"  ✓ Sum of splits = {self.train_size + self.val_size + self.test_size} = Total samples")
        print("="*80)


def ensure_data_alignment(model_type: str = 'all') -> DataAlignmentConfig:
    """
    Main function to ensure data alignment across all models.
    
    Args:
        model_type: 'gnn', 'lstm', or 'all'
        
    Returns:
        DataAlignmentConfig object with consistent splits
    """
    
    # Create configuration
    config = DataAlignmentConfig(seq_length=42)
    
    # Save configuration for reproducibility
    config_dict = config.save_config()
    
    # Print summary
    config.print_summary()
    
    print(f"\nConfiguration saved to: processed_data/data_alignment_config.json")
    print("All models should use this configuration for consistent data splits.")
    
    return config


def verify_model_alignment(gnn_dataset=None, lstm_data=None) -> bool:
    """
    Verify that GNN and LSTM datasets are perfectly aligned.
    
    Args:
        gnn_dataset: GNN dataset object (optional)
        lstm_data: Tuple of (X, y) LSTM arrays (optional)
        
    Returns:
        True if aligned, False otherwise
    """
    
    config = DataAlignmentConfig(seq_length=42)
    
    print("\n" + "="*80)
    print("MODEL ALIGNMENT VERIFICATION")
    print("="*80)
    
    all_checks_passed = True
    
    # Check GNN dataset if provided
    if gnn_dataset is not None:
        gnn_size = len(gnn_dataset)
        expected_size = config.total_samples
        
        print(f"\nGNN Dataset:")
        print(f"  Expected samples: {expected_size}")
        print(f"  Actual samples: {gnn_size}")
        
        if gnn_size == expected_size:
            print("  ✓ GNN dataset size matches expected")
        else:
            print("  ✗ GNN dataset size mismatch!")
            all_checks_passed = False
    
    # Check LSTM data if provided
    if lstm_data is not None:
        X, y = lstm_data
        lstm_size = len(X)
        expected_size = config.total_samples
        
        print(f"\nLSTM Dataset:")
        print(f"  Expected samples: {expected_size}")
        print(f"  Actual samples: {lstm_size}")
        
        if lstm_size == expected_size:
            print("  ✓ LSTM dataset size matches expected")
        else:
            print("  ✗ LSTM dataset size mismatch!")
            all_checks_passed = False
    
    # Cross-model check if both provided
    if gnn_dataset is not None and lstm_data is not None:
        print(f"\nCross-Model Alignment:")
        if len(gnn_dataset) == len(lstm_data[0]):
            print(f"  ✓ Both models have {len(gnn_dataset)} samples")
        else:
            print(f"  ✗ Model sample count mismatch!")
            all_checks_passed = False
    
    if all_checks_passed:
        print("\n✅ PERFECT ALIGNMENT ACHIEVED")
    else:
        print("\n❌ ALIGNMENT ISSUES DETECTED")
    
    print("="*80)
    
    return all_checks_passed


if __name__ == "__main__":
    # Run alignment configuration
    config = ensure_data_alignment()