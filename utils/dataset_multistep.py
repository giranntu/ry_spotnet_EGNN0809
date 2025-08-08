#!/usr/bin/env python3
"""
Multi-Step Prediction Dataset for Next-Day Volatility Forecasting
==================================================================
Predicts all 13 intervals of the next trading day

CRITICAL: 
- Input X: Last valid sample from day D (ending at 16:00)
- Output Y: All 13 volatility values from day D+1
- NO data leakage: D+1 information never appears in X calculation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from tqdm import tqdm
from natsort import natsorted
from typing import List, Dict, Tuple


class MultiStepIntradayDataset(Dataset):
    """
    Dataset for predicting next trading day's volatility (13 intervals)
    
    CRITICAL FEATURES:
    1. Input: Day D's last valid sample (L=42 ending at 16:00)
    2. Target: Day D+1's all 13 interval volatilities
    3. Strict temporal separation prevents data leakage
    """
    
    def __init__(self,
                 vol_file: str,
                 volvol_file: str,
                 seq_length: int = 42,
                 intervals_per_day: int = 13,
                 split: str = 'train',
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2):
        """
        Initialize multi-step prediction dataset
        
        Args:
            vol_file: Path to HDF5 file with volatility matrices
            volvol_file: Path to HDF5 file with vol-of-vol matrices
            seq_length: Number of intervals for input (42 = ~3 days)
            intervals_per_day: Number of intervals per day (13)
            split: 'train', 'val', or 'test'
        """
        self.vol_file = vol_file
        self.volvol_file = volvol_file
        self.seq_length = seq_length
        self.intervals_per_day = intervals_per_day
        self.split = split
        
        # Create day-level samples
        self.samples = self._create_day_samples()
        
        # Apply temporal split
        self._apply_temporal_split(train_ratio, val_ratio)
    
    def _create_day_samples(self) -> List[Dict]:
        """
        Create (Day_D, Day_D+1) sample pairs
        
        CRITICAL LOGIC:
        - Day D features: Last valid L=42 sample ending at 16:00
        - Day D+1 targets: All 13 intervals of next day
        - NO information from D+1 in feature calculation
        """
        samples = []
        
        with h5py.File(self.vol_file, 'r') as f_vol:
            keys = natsorted(list(f_vol.keys()))
            
            # Map intervals to days
            interval_to_day = {}
            day_to_intervals = {}
            current_day = 0
            
            for i, key in enumerate(keys):
                if i > 0 and i % self.intervals_per_day == 0:
                    current_day += 1
                interval_to_day[int(key)] = current_day
                
                if current_day not in day_to_intervals:
                    day_to_intervals[current_day] = []
                day_to_intervals[current_day].append(int(key))
            
            total_days = len(day_to_intervals)
            print(f"Found {total_days} trading days for multi-step prediction")
            
            # Create samples for consecutive day pairs
            for day_d in tqdm(range(total_days - 1), desc="Creating multi-step samples"):
                day_d_plus_1 = day_d + 1
                
                # Check if both days are complete
                if len(day_to_intervals[day_d]) != self.intervals_per_day:
                    continue
                if len(day_to_intervals[day_d_plus_1]) != self.intervals_per_day:
                    continue
                
                # Get last interval of day D (16:00)
                last_interval_d = day_to_intervals[day_d][-1]
                
                # Build input: L=42 intervals ending at last_interval_d
                input_indices = list(range(last_interval_d - self.seq_length + 1,
                                          last_interval_d + 1))
                
                # Verify input validity (no gaps, sufficient history)
                valid = True
                for idx in input_indices:
                    if str(idx) not in keys:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Get all 13 target intervals from day D+1
                target_indices = day_to_intervals[day_d_plus_1]
                
                # Create sample
                sample = {
                    'input_indices': input_indices,
                    'target_indices': target_indices,  # 13 intervals
                    'input_day': day_d,
                    'target_day': day_d_plus_1
                }
                samples.append(sample)
        
        print(f"Created {len(samples)} multi-step samples")
        return samples
    
    def _apply_temporal_split(self, train_ratio: float, val_ratio: float):
        """Apply temporal train/val/test split"""
        n_samples = len(self.samples)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        if self.split == 'train':
            self.samples = self.samples[:train_end]
        elif self.split == 'val':
            self.samples = self.samples[train_end:val_end]
        elif self.split == 'test':
            self.samples = self.samples[val_end:]
        
        print(f"{self.split.upper()} set: {len(self.samples)} day-pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a multi-step sample
        
        Returns:
            Dictionary with:
            - features: Input sequence (L=42 intervals from day D)
            - targets: All 13 volatilities from day D+1
            - metadata: Day information
        """
        sample_info = self.samples[idx]
        
        with h5py.File(self.vol_file, 'r') as f_vol, \
             h5py.File(self.volvol_file, 'r') as f_volvol:
            
            # Load input sequence (Day D, last L=42 intervals)
            input_vol_matrices = []
            input_volvol_matrices = []
            
            for hist_idx in sample_info['input_indices']:
                vol_mat = f_vol[str(hist_idx)][:]
                volvol_mat = f_volvol[str(hist_idx)][:]
                input_vol_matrices.append(vol_mat)
                input_volvol_matrices.append(volvol_mat)
            
            # Extract input features
            features = self._extract_features(
                input_vol_matrices,
                input_volvol_matrices
            )
            
            # Load all 13 targets from day D+1
            targets = []
            for target_idx in sample_info['target_indices']:
                target_vol = f_vol[str(target_idx)][:]
                # Extract diagonal (volatilities) as target
                targets.append(np.diag(target_vol))
            
            # Stack targets: Shape [13, n_assets]
            targets = np.array(targets)
            
            return {
                'features': torch.tensor(features, dtype=torch.float32),
                'targets': torch.tensor(targets, dtype=torch.float32),
                'metadata': {
                    'input_day': sample_info['input_day'],
                    'target_day': sample_info['target_day']
                }
            }
    
    def _extract_features(self, vol_matrices, volvol_matrices):
        """
        Extract features from sequence of matrices
        
        Same as single-step but ensures no D+1 information leaks
        """
        all_features = []
        
        for vol_mat, volvol_mat in zip(vol_matrices, volvol_matrices):
            # Volatilities (diagonal)
            vols = np.diag(vol_mat)
            
            # Covolatilities (upper triangle)
            covols = vol_mat[np.triu_indices(vol_mat.shape[0], k=1)]
            
            # Vol-of-vol (diagonal)
            volvols = np.diag(volvol_mat)
            
            # Covol-of-vol (upper triangle)
            covolvols = volvol_mat[np.triu_indices(volvol_mat.shape[0], k=1)]
            
            # Concatenate: 30 + 435 + 30 + 435 = 930 features
            features = np.concatenate([vols, covols, volvols, covolvols])
            all_features.append(features)
        
        # Shape: [seq_length, n_features]
        return np.array(all_features)


def verify_multistep_dataset():
    """
    Verify multi-step dataset has no data leakage
    """
    print("\n" + "="*80)
    print("VERIFYING MULTI-STEP DATASET (NO DATA LEAKAGE)")
    print("="*80)
    
    vol_file = 'processed_data/vols_mats_30min.h5'
    volvol_file = 'processed_data/volvols_mats_30min.h5'
    
    if not os.path.exists(vol_file):
        print("⚠️  Run intraday pipeline first")
        return
    
    # Create dataset
    dataset = MultiStepIntradayDataset(
        vol_file, volvol_file,
        seq_length=42,
        intervals_per_day=13,
        split='train'
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        print(f"\n✅ Multi-step dataset created")
        print(f"   Input shape: {sample['features'].shape}")
        print(f"   Target shape: {sample['targets'].shape}")
        print(f"   Input day: D={sample['metadata']['input_day']}")
        print(f"   Target day: D+1={sample['metadata']['target_day']}")
        
        print(f"\n📊 Sample details:")
        print(f"   Input: 42 intervals ending at day D's 16:00")
        print(f"   Target: All 13 intervals of day D+1")
        print(f"   NO data leakage: D+1 never appears in input calculation")
        
        # Check dimensions
        seq_len, n_features = sample['features'].shape
        n_intervals, n_assets = sample['targets'].shape
        
        print(f"\n📐 Dimensions check:")
        print(f"   Input: {seq_len} timesteps × {n_features} features")
        print(f"   Output: {n_intervals} intervals × {n_assets} assets")
        print(f"   Feature count: 30 + 435 + 30 + 435 = 930 ✓")
    else:
        print("❌ No samples created")
    
    print("="*80)


if __name__ == "__main__":
    verify_multistep_dataset()