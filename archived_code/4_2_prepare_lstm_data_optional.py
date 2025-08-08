#!/usr/bin/env python3
"""
Prepare data for LSTM training from the standardized HDF5 files
"""

import numpy as np
import h5py
import os
from tqdm import tqdm

def prepare_lstm_data(seq_length=42):
    """
    Convert the HDF5 matrix data to LSTM format (sequences)
    """
    
    # Create output directory
    output_dir = f'processed_data/cached_lstm_vols_mats_taq_{seq_length}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load standardized volatility data
    print(f"Loading standardized volatility data...")
    with h5py.File('processed_data/vols_mats_taq_standardized.h5', 'r') as f:
        # Get all keys and sort them numerically
        keys = sorted(list(f.keys()), key=int)
        num_samples = len(keys)
        
        # Get shape from first matrix
        first_matrix = f[keys[0]][:]
        n_assets = first_matrix.shape[0]
        
        print(f"Found {num_samples} time points with {n_assets} assets")
        
        # Prepare sequences
        X_list = []
        y_list = []
        
        print(f"Creating sequences with length {seq_length}...")
        for i in tqdm(range(num_samples - seq_length)):
            # Create input sequence: seq_length matrices
            sequence = []
            for j in range(seq_length):
                matrix = f[keys[i+j]][:]
                # Extract diagonal (volatilities) as features
                diag_features = np.diag(matrix)
                sequence.append(diag_features)
            
            # Stack to create sequence matrix (seq_length x n_assets)
            X = np.array(sequence)
            
            # Target: next time step volatilities (diagonal of next matrix)
            y = np.diag(f[keys[i+seq_length]][:])
            
            X_list.append(X)
            y_list.append(y)
    
    # Convert to numpy arrays
    X_matrices = np.array(X_list)
    y_vectors = np.array(y_list)
    
    print(f"Created dataset with shapes:")
    print(f"  X (input sequences): {X_matrices.shape}")
    print(f"  y (target vectors): {y_vectors.shape}")
    
    # Save the prepared data
    np.save(os.path.join(output_dir, 'x_matrices.npy'), X_matrices)
    np.save(os.path.join(output_dir, 'y_x_vectors.npy'), y_vectors)
    
    print(f"Data saved to {output_dir}/")
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  X range: [{X_matrices.min():.4f}, {X_matrices.max():.4f}]")
    print(f"  y range: [{y_vectors.min():.4f}, {y_vectors.max():.4f}]")
    print(f"  X mean: {X_matrices.mean():.4f}")
    print(f"  y mean: {y_vectors.mean():.4f}")
    
    return X_matrices, y_vectors

if __name__ == "__main__":
    prepare_lstm_data(seq_length=42)