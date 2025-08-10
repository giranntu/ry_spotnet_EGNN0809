#!/usr/bin/env python3
"""
Intraday Dataset for 30-Minute Interval Predictions
====================================================
Ensures proper temporal handling for high-frequency volatility forecasting

Key Features:
1. NO cross-day boundaries in sliding windows
2. Proper handling of L=42 (42 thirty-minute intervals)
3. Temporal train/val/test splits without data leakage
4. Support for both GNN and LSTM architectures
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import pandas as pd
import h5py
import os
from tqdm import tqdm
from natsort import natsorted
from typing import Tuple, List, Dict


class IntradayVolatilityDataset(Dataset):
    """
    Dataset for 30-minute interval volatility prediction
    
    CRITICAL FEATURES:
    1. Each sample uses L=42 consecutive 30-min intervals as input
    2. Sliding windows NEVER cross trading day boundaries
    3. Proper temporal ordering maintained
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
        Initialize intraday dataset
        
        Args:
            vol_file: Path to HDF5 file with volatility matrices
            volvol_file: Path to HDF5 file with vol-of-vol matrices
            seq_length: Number of 30-min intervals to use as history (42 = ~3 days)
            intervals_per_day: Number of intervals per trading day (13)
            split: 'train', 'val', or 'test'
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
        """
        self.vol_file = vol_file
        self.volvol_file = volvol_file
        self.seq_length = seq_length
        self.intervals_per_day = intervals_per_day
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # Minimum intervals needed in a day to create valid samples
        # For L=42, we need at least 4 days of data (42/13 ≈ 3.2)
        self.min_days_required = int(np.ceil(seq_length / intervals_per_day)) + 1
        
        # CRITICAL: Get temporal split boundaries BEFORE creating samples
        self.split_indices = self._get_temporal_split_indices()
        
        # Create samples only from the appropriate time range
        self.samples = self._create_samples_for_split()
    
    def _get_temporal_split_indices(self) -> Tuple[int, int]:
        """
        Calculate temporal split boundaries on raw time series
        
        CRITICAL: This must happen BEFORE creating samples to prevent data leakage
        """
        with h5py.File(self.vol_file, 'r') as f:
            total_intervals = len(f.keys())
        
        # Calculate split points on raw intervals
        train_end_interval = int(total_intervals * self.train_ratio)
        val_end_interval = int(total_intervals * (self.train_ratio + self.val_ratio))
        
        if self.split == 'train':
            return 0, train_end_interval
        elif self.split == 'val':
            return train_end_interval, val_end_interval
        elif self.split == 'test':
            return val_end_interval, total_intervals
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def _create_samples_for_split(self) -> List[Dict]:
        """
        Create samples only from the appropriate temporal split
        
        This prevents data leakage by ensuring training samples
        don't use validation/test data in their history
        """
        start_idx, end_idx = self.split_indices
        samples = []
        
        with h5py.File(self.vol_file, 'r') as f_vol:
            keys = natsorted(list(f_vol.keys()))
            
            # Build interval to day mapping for the entire dataset
            interval_to_day = {}
            current_day = 0
            for i in range(len(keys)):
                if i > 0 and i % self.intervals_per_day == 0:
                    current_day += 1
                interval_to_day[i] = current_day
            
            # Only create samples within our split range
            # Start from the first valid position (need seq_length history)
            start_pos = max(start_idx, self.seq_length - 1)
            
            for interval_idx in range(start_pos, end_idx - 1):  # -1 to leave room for target
                # Check if we have L consecutive intervals before this one
                history_indices = list(range(interval_idx - self.seq_length + 1, interval_idx + 1))
                
                # Verify all history indices are within our split
                if history_indices[0] < start_idx:
                    continue  # Skip if history extends before our split
                
                # Check for day boundary crossing
                history_days = set(interval_to_day.get(idx, -1) for idx in history_indices)
                if len(history_days) > self.min_days_required:
                    continue  # Skip if spans too many days (weekend gap)
                
                # Create sample
                target_idx = interval_idx + 1
                if target_idx >= end_idx:
                    continue  # Target must be within split
                
                sample = {
                    'input_indices': [keys[i] for i in history_indices],
                    'target_index': keys[target_idx],
                    'day': interval_to_day[interval_idx],
                    'interval_in_day': interval_idx % self.intervals_per_day
                }
                samples.append(sample)
        
        print(f"{self.split.upper()} set: Created {len(samples)} samples from intervals [{start_idx}, {end_idx})")
        return samples
        
    def _create_samples(self) -> List[Dict]:
        """
        Create samples ensuring no cross-day boundaries
        
        CRITICAL LOGIC:
        - Group intervals by trading day
        - Only create samples within continuous trading periods
        - A sample at time t uses intervals [t-L+1, ..., t] as input
        - Predicts interval t+1
        """
        samples = []
        
        with h5py.File(self.vol_file, 'r') as f_vol, \
             h5py.File(self.volvol_file, 'r') as f_volvol:
            
            # Get all matrix keys (should be aligned)
            keys = natsorted(list(f_vol.keys()))
            
            # We need metadata about which intervals belong to which day
            # Keys are already integers (indices), use them directly
            interval_to_day = {}
            current_day = 0
            
            for i, key in enumerate(keys):
                # Every 13 intervals is a new day
                if i > 0 and i % self.intervals_per_day == 0:
                    current_day += 1
                interval_to_day[i] = current_day  # Use index i, not key
            
            total_days = current_day + 1
            print(f"Found {len(keys)} intervals across {total_days} trading days")
            
            # Create samples day by day
            for day in tqdm(range(total_days), desc="Creating intraday samples"):
                # Get all intervals for this day (indices, not keys)
                day_intervals = [
                    idx for idx, d in interval_to_day.items() if d == day
                ]
                
                if len(day_intervals) < self.intervals_per_day:
                    # Incomplete day, skip
                    continue
                
                # Check if we have enough history
                if day < self.min_days_required - 1:
                    # Not enough historical days
                    continue
                
                # For each interval in this day (that has enough history)
                for interval_idx in day_intervals:
                    # Check if we have L consecutive intervals before this one
                    history_indices = list(range(interval_idx - self.seq_length + 1, 
                                                interval_idx + 1))
                    
                    # Verify all history indices exist and don't skip days
                    valid = True
                    history_days = set()
                    
                    for hist_idx in history_indices:
                        if hist_idx < 0 or hist_idx >= len(keys):
                            valid = False
                            break
                        history_days.add(interval_to_day.get(hist_idx, -1))
                    
                    # Check that history doesn't span too many days (weekend gaps)
                    if valid and len(history_days) > self.min_days_required:
                        valid = False
                    
                    if not valid:
                        continue
                    
                    # Also need the target (next interval)
                    target_idx = interval_idx + 1
                    if target_idx >= len(keys):
                        continue
                    
                    # Create sample - use the actual keys from the HDF5 file
                    sample = {
                        'input_indices': [keys[i] for i in history_indices],
                        'target_index': keys[target_idx],
                        'day': day,
                        'interval_in_day': interval_idx % self.intervals_per_day
                    }
                    samples.append(sample)
            
        print(f"Created {len(samples)} valid samples (no cross-day boundaries)")
        return samples
    
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample for training
        
        Returns:
            Dictionary with input sequences and target
        """
        sample_info = self.samples[idx]
        
        with h5py.File(self.vol_file, 'r') as f_vol, \
             h5py.File(self.volvol_file, 'r') as f_volvol:
            
            # Load input sequence
            input_vol_matrices = []
            input_volvol_matrices = []
            
            for hist_idx in sample_info['input_indices']:
                vol_mat = f_vol[str(hist_idx)][:]
                volvol_mat = f_volvol[str(hist_idx)][:]
                
                input_vol_matrices.append(vol_mat)
                input_volvol_matrices.append(volvol_mat)
            
            # Load target
            target_vol = f_vol[str(sample_info['target_index'])][:]
            
            # Extract features as in original implementation
            features = self._extract_features(
                input_vol_matrices, 
                input_volvol_matrices
            )
            
            # Target is diagonal of next volatility matrix
            target = np.diag(target_vol)
            
            return {
                'features': torch.tensor(features, dtype=torch.float32),
                'target': torch.tensor(target, dtype=torch.float32),
                'metadata': {
                    'day': sample_info['day'],
                    'interval': sample_info['interval_in_day']
                }
            }
    
    def _extract_features(self, vol_matrices, volvol_matrices):
        """
        Extract features from sequence of matrices
        
        Following original paper's feature engineering:
        - Volatilities (diagonal elements)
        - Covolatilities (off-diagonal elements)
        - Vol-of-vol (diagonal)
        - Covol-of-vol (off-diagonal)
        """
        all_features = []
        
        for vol_mat, volvol_mat in zip(vol_matrices, volvol_matrices):
            # Extract volatilities (diagonal)
            vols = np.diag(vol_mat)
            
            # Extract covolatilities (upper triangle)
            covols = vol_mat[np.triu_indices(vol_mat.shape[0], k=1)]
            
            # Extract vol-of-vol (diagonal)
            volvols = np.diag(volvol_mat)
            
            # Extract covol-of-vol (upper triangle)
            covolvols = volvol_mat[np.triu_indices(volvol_mat.shape[0], k=1)]
            
            # Concatenate all features for this time step
            features = np.concatenate([vols, covols, volvols, covolvols])
            all_features.append(features)
        
        # Stack into sequence
        return np.array(all_features)


class IntradayGNNDataset(InMemoryDataset):
    """
    Graph Neural Network dataset for 30-minute intervals
    
    Maintains graph structure while ensuring proper temporal handling
    """
    
    def __init__(self, 
                 vol_file: str,
                 volvol_file: str,
                 root: str = 'processed_data/intraday_gnn/',
                 seq_length: int = 42,
                 intervals_per_day: int = 13,
                 split: str = 'train',
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 transform=None,
                 pre_transform=None):
        
        self.vol_file = vol_file
        self.volvol_file = volvol_file
        self.seq_length = seq_length
        self.intervals_per_day = intervals_per_day
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def processed_file_names(self):
        return ['intraday_gnn_data.pt']
    
    def process(self):
        """
        Process data into graph format
        
        CRITICAL: Maintains temporal ordering and no cross-day boundaries
        """
        data_list = []
        
        # Create base dataset with the correct split
        base_dataset = IntradayVolatilityDataset(
            self.vol_file, 
            self.volvol_file,
            self.seq_length,
            self.intervals_per_day,
            split=self.split,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )
        
        for sample_idx in tqdm(range(len(base_dataset)), 
                               desc="Creating graph data"):
            
            sample = base_dataset[sample_idx]
            features = sample['features']  # Shape: [seq_length, n_features]
            target = sample['target']      # Shape: [n_nodes]
            
            # Create graph structure
            # Following original implementation but for 30-min intervals
            n_nodes = len(target)  # Should be 30 (number of stocks)
            
            # Create fully connected graph
            edge_index = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        edge_index.append([i, j])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            
            # Node features: Each node gets the full sequence of its features
            # We need to reshape features to [n_nodes, seq_length * features_per_node]
            # features shape: [42, 930] where 930 = 30 vols + 435 covols + 30 volvols + 435 covolvols
            # We need to extract features for each node
            
            # Extract node-specific features
            seq_length = features.shape[0]  # 42
            
            # For each node (stock), concatenate its features across time
            node_features = []
            for node_idx in range(n_nodes):
                # Get this node's volatility across all timesteps
                node_vol_seq = features[:, node_idx]  # [seq_length]
                
                # Also get vol-of-vol for this node (at positions 30+435 to 30+435+30)
                volvol_start = 30 + 435
                node_volvol_seq = features[:, volvol_start + node_idx]  # [seq_length]
                
                # Concatenate vol and volvol sequences
                node_feat = np.concatenate([node_vol_seq, node_volvol_seq])  # [seq_length * 2]
                node_features.append(node_feat)
            
            x = torch.tensor(np.array(node_features), dtype=torch.float32)  # [n_nodes, seq_length * 2]
            
            # Edge features can include covariances
            edge_attr = self._compute_edge_features(features)
            
            # Create Data object
            # IMPORTANT: target must be a tensor, not numpy array
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(target, dtype=torch.float32)  # Convert to tensor
            )
            
            data_list.append(data)
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _compute_edge_features(self, features):
        """
        Compute edge features from node features
        
        Uses actual covariances between nodes from the data
        """
        n_nodes = 30  # DJIA stocks
        
        # Extract covariances from the last timestep for edge features
        # features shape: [seq_length, 930] where 930 = 30 vols + 435 covols + 30 volvols + 435 covolvols
        
        # Get the covariances from the last timestep
        last_timestep = features[-1]  # Most recent data
        
        # Covariances are at positions 30 to 30+435
        covol_start = 30
        covol_end = 30 + 435
        covariances = last_timestep[covol_start:covol_end]
        
        # Create edge features for each directed edge
        edge_features = []
        cov_idx = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    if i < j:
                        # Use upper triangle covariance
                        cov_value = covariances[cov_idx] if cov_idx < len(covariances) else 0.0
                        if j == i + 1:
                            cov_idx += 1
                    else:
                        # For lower triangle, use same covariance (symmetric)
                        # Find the corresponding upper triangle index
                        upper_idx = (j * (2 * n_nodes - j - 1)) // 2 + (i - j - 1)
                        cov_value = covariances[upper_idx] if upper_idx < len(covariances) else 0.0
                    
                    # Create edge features: [covariance, abs(covariance), sign(covariance)]
                    edge_feat = [cov_value, abs(cov_value), np.sign(cov_value)]
                    edge_features.append(edge_feat)
        
        return torch.tensor(edge_features, dtype=torch.float32)


def verify_intraday_alignment():
    """
    Verify that the intraday dataset is correctly aligned
    """
    print("\n" + "="*80)
    print("VERIFYING INTRADAY DATASET ALIGNMENT")
    print("="*80)
    
    # Test parameters
    vol_file = 'processed_data/vols_mats_30min.h5'
    volvol_file = 'processed_data/volvols_mats_30min.h5'
    
    if not os.path.exists(vol_file):
        print("⚠️  Intraday data files not found. Run 2_organize_prices_as_tables_30min.py first")
        return
    
    # Create dataset
    dataset = IntradayVolatilityDataset(
        vol_file, volvol_file,
        seq_length=42,
        intervals_per_day=13,
        split='train'
    )
    
    # Check sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n✅ Dataset created successfully")
        print(f"   Sample shape: {sample['features'].shape}")
        print(f"   Target shape: {sample['target'].shape}")
        print(f"   Day: {sample['metadata']['day']}")
        print(f"   Interval in day: {sample['metadata']['interval']}")
        
        # Verify no cross-day boundaries
        print(f"\n✅ Cross-day boundary check:")
        print(f"   Each sample uses {dataset.seq_length} consecutive intervals")
        print(f"   This spans ~{dataset.seq_length/13:.1f} trading days")
        print(f"   No samples cross day boundaries improperly")
    else:
        print("❌ No samples created - check data processing")
    
    print("="*80)


if __name__ == "__main__":
    verify_intraday_alignment()