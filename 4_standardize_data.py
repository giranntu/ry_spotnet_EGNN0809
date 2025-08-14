#!/usr/bin/env python3
"""
Volatility Standardization with Log-Transform
==============================================
Properly handles positive volatility values while normalizing for neural networks

Key features:
1. Volatilities (diagonal) use log-transform to preserve positivity
2. Covariances (off-diagonal) use standard normalization
3. Maintains physical interpretability of volatility
"""

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VolatilityStandardizer:
    """
    Proper standardization for volatility matrices that preserves positivity
    """
    
    def __init__(self, method='log_transform'):
        """
        Initialize standardizer
        
        Args:
            method: Standardization method for volatilities
                - 'log_transform': Take log then standardize (recommended)
                - 'scale_only': Divide by std without centering
                - 'robust_scale': Use median/IQR scaling
                - 'minmax': Scale to [0.1, 1] range
        """
        self.method = method
        self.vol_scaler = None
        self.cov_scaler = None
        self.fitted = False
        
    def fit(self, matrices, matrix_names=None):
        """
        Fit standardization parameters on training data only
        
        Args:
            matrices: List of covariance matrices
            matrix_names: Optional list of matrix identifiers
        """
        print(f"Fitting standardizer using method: {self.method}")
        
        # Collect all diagonal and off-diagonal elements
        all_diagonals = []
        all_off_diagonals = []
        
        for matrix in matrices:
            # Extract diagonal (volatilities - must be positive)
            diagonal = np.diag(matrix)
            
            # Validate positivity
            if np.any(diagonal <= 0):
                print(f"Warning: Found non-positive volatility values: min={diagonal.min()}")
                diagonal = np.maximum(diagonal, 1e-8)  # Enforce minimum value
            
            all_diagonals.append(diagonal)
            
            # Extract off-diagonal (covariances - can be negative)
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            off_diagonal = matrix[mask]
            all_off_diagonals.append(off_diagonal)
        
        all_diagonals = np.concatenate(all_diagonals)
        all_off_diagonals = np.concatenate(all_off_diagonals)
        
        # Fit volatility scaler based on method
        if self.method == 'log_transform':
            # Log-transform volatilities then standardize
            log_vols = np.log(all_diagonals)
            self.vol_scaler = StandardScaler()
            self.vol_scaler.fit(log_vols.reshape(-1, 1))
            self.vol_mean = np.mean(all_diagonals)
            self.vol_std = np.std(all_diagonals)
            print(f"  Log-volatility stats: mean={log_vols.mean():.4f}, std={log_vols.std():.4f}")
            
        elif self.method == 'scale_only':
            # Scale by std without centering (preserves positivity)
            self.vol_mean = np.mean(all_diagonals)
            self.vol_std = np.std(all_diagonals)
            print(f"  Volatility stats: mean={self.vol_mean:.4f}, std={self.vol_std:.4f}")
            
        elif self.method == 'robust_scale':
            # Use median and IQR for robust scaling
            self.vol_median = np.median(all_diagonals)
            self.vol_q1 = np.percentile(all_diagonals, 25)
            self.vol_q3 = np.percentile(all_diagonals, 75)
            self.vol_iqr = self.vol_q3 - self.vol_q1
            print(f"  Volatility stats: median={self.vol_median:.4f}, IQR={self.vol_iqr:.4f}")
            
        elif self.method == 'minmax':
            # Scale to [0.1, 1] range
            self.vol_min = np.min(all_diagonals)
            self.vol_max = np.max(all_diagonals)
            print(f"  Volatility range: [{self.vol_min:.4f}, {self.vol_max:.4f}]")
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit covariance scaler (standard normalization is OK for covariances)
        self.cov_scaler = StandardScaler()
        self.cov_scaler.fit(all_off_diagonals.reshape(-1, 1))
        print(f"  Covariance stats: mean={self.cov_scaler.mean_[0]:.6f}, std={self.cov_scaler.scale_[0]:.6f}")
        
        self.fitted = True
        
    def transform(self, matrix):
        """
        Transform a single covariance matrix
        
        Args:
            matrix: Covariance matrix to transform
            
        Returns:
            Transformed matrix
        """
        if not self.fitted:
            raise ValueError("Standardizer must be fitted before transform")
        
        n = matrix.shape[0]
        transformed = np.zeros_like(matrix)
        
        # Transform diagonal (volatilities)
        diagonal = np.diag(matrix)
        diagonal = np.maximum(diagonal, 1e-8)  # Ensure positivity
        
        if self.method == 'log_transform':
            # Log-transform then standardize
            log_diagonal = np.log(diagonal)
            standardized_diagonal = self.vol_scaler.transform(log_diagonal.reshape(-1, 1)).flatten()
            
        elif self.method == 'scale_only':
            # Scale by std without centering
            standardized_diagonal = diagonal / self.vol_std
            
        elif self.method == 'robust_scale':
            # Robust scaling using median and IQR
            standardized_diagonal = (diagonal - self.vol_median) / (self.vol_iqr + 1e-8)
            
        elif self.method == 'minmax':
            # Min-max scaling to [0.1, 1]
            standardized_diagonal = 0.1 + 0.9 * (diagonal - self.vol_min) / (self.vol_max - self.vol_min + 1e-8)
        
        # Fill diagonal
        np.fill_diagonal(transformed, standardized_diagonal)
        
        # Transform off-diagonal (covariances) - standard normalization
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = matrix[mask]
        standardized_off_diagonal = self.cov_scaler.transform(off_diagonal.reshape(-1, 1)).flatten()
        transformed[mask] = standardized_off_diagonal
        
        return transformed
    
    def inverse_transform_diagonal(self, standardized_diagonal):
        """
        Inverse transform diagonal elements back to original volatility scale
        
        Args:
            standardized_diagonal: Standardized diagonal values
            
        Returns:
            Original scale volatilities (always positive)
        """
        if not self.fitted:
            raise ValueError("Standardizer must be fitted before inverse transform")
        
        if self.method == 'log_transform':
            # Inverse standardize then exp
            log_diagonal = self.vol_scaler.inverse_transform(standardized_diagonal.reshape(-1, 1)).flatten()
            original = np.exp(log_diagonal)
            
        elif self.method == 'scale_only':
            # Multiply by std
            original = standardized_diagonal * self.vol_std
            
        elif self.method == 'robust_scale':
            # Inverse robust scaling
            original = standardized_diagonal * self.vol_iqr + self.vol_median
            
        elif self.method == 'minmax':
            # Inverse min-max scaling
            standardized_01 = (standardized_diagonal - 0.1) / 0.9
            original = standardized_01 * (self.vol_max - self.vol_min) + self.vol_min
        
        # Ensure positivity
        return np.maximum(original, 1e-8)


def define_time_splits(total_matrices):
    """
    Define train/validation/test splits based on actual data length
    
    For ~2000 matrices (2019-2025 data):
    Train: First 1008 matrices (~50%) - 2019-2022
    Validation: Next 252 matrices (~13%) - 2023
    Test: Remaining matrices (~37%) - 2024-2025
    """
    train_end_idx = min(1008, int(total_matrices * 0.5))
    val_end_idx = min(1260, train_end_idx + 252)
    
    return train_end_idx, val_end_idx


def standardize_matrices(file_type='vols', method='log_transform', suffix='30min'):
    """
    Standardize volatility matrices preserving positivity
    
    Args:
        file_type: 'vols' or 'volvols'
        method: Standardization method for volatilities
        suffix: Data suffix ('30min' for intraday, 'taq' for daily)
    """
    print(f"\n{'='*60}")
    print(f"Standardizing {file_type} matrices with method: {method}")
    print(f"{'='*60}")
    
    # Define file paths
    input_file_path = f'processed_data/{file_type}_mats_{suffix}.h5'
    output_file_path = f'processed_data/{file_type}_mats_{suffix}_standardized.h5'
    mean_std_csv_path = f'processed_data/{file_type}_{suffix}_mean_std_scalers.csv'
    
    # Load all matrices
    matrices = []
    matrix_names = []
    
    with h5py.File(input_file_path, 'r') as input_file:
        sorted_keys = sorted(input_file.keys(), key=int)
        
        for dataset_name in sorted_keys:
            matrix = input_file[dataset_name][:]
            matrices.append(matrix)
            matrix_names.append(dataset_name)
    
    print(f"Loaded {len(matrices)} matrices")
    
    # Check for negative values in raw data
    negative_count = 0
    for i, matrix in enumerate(matrices):
        diag = np.diag(matrix)
        if np.any(diag <= 0):
            negative_count += 1
            print(f"  Warning: Matrix {i} has non-positive diagonal values: min={diag.min():.6f}")
    
    if negative_count > 0:
        print(f"  Found {negative_count} matrices with non-positive volatilities - will correct")
    
    # Define temporal splits
    train_end_idx, val_end_idx = define_time_splits(len(matrices))
    print(f"\nTemporal splits:")
    print(f"  Train: 0-{train_end_idx} ({train_end_idx} matrices)")
    print(f"  Val: {train_end_idx}-{val_end_idx} ({val_end_idx-train_end_idx} matrices)")
    print(f"  Test: {val_end_idx}-{len(matrices)} ({len(matrices)-val_end_idx} matrices)")
    
    # Fit standardizer on training data only
    train_matrices = matrices[:train_end_idx]
    standardizer = VolatilityStandardizer(method=method)
    standardizer.fit(train_matrices)
    
    # Transform all matrices
    print("\nTransforming matrices...")
    standardized_matrices = []
    
    for i, matrix in enumerate(matrices):
        standardized = standardizer.transform(matrix)
        standardized_matrices.append(standardized)
        
        # Verify no NaN values
        if np.any(np.isnan(standardized)):
            print(f"  Warning: NaN values in standardized matrix {i}")
    
    # Verify transformation preserves structure
    print("\nVerification:")
    for i in range(min(5, len(matrices))):
        orig_diag = np.diag(matrices[i])
        std_diag = np.diag(standardized_matrices[i])
        
        if method == 'log_transform':
            # Check that we can recover approximate original values
            recovered = standardizer.inverse_transform_diagonal(std_diag)
            error = np.mean(np.abs(recovered - orig_diag))
            print(f"  Matrix {i}: Original vol mean={orig_diag.mean():.4f}, "
                  f"Standardized mean={std_diag.mean():.4f}, Recovery error={error:.6f}")
        else:
            print(f"  Matrix {i}: Original vol mean={orig_diag.mean():.4f}, "
                  f"Standardized mean={std_diag.mean():.4f}, min={std_diag.min():.4f}")
    
    # Save standardized matrices
    print("\nSaving standardized matrices...")
    with h5py.File(output_file_path, 'w') as output_file:
        for i, matrix_name in enumerate(matrix_names):
            output_file.create_dataset(matrix_name, data=standardized_matrices[i])
    
    # Save standardization parameters
    if method == 'log_transform':
        params_df = pd.DataFrame({
            'Method': [method],
            'Vol_LogMean': [standardizer.vol_scaler.mean_[0]],
            'Vol_LogStd': [standardizer.vol_scaler.scale_[0]],
            'Vol_OrigMean': [standardizer.vol_mean],
            'Vol_OrigStd': [standardizer.vol_std],
            'Cov_Mean': [standardizer.cov_scaler.mean_[0]],
            'Cov_Std': [standardizer.cov_scaler.scale_[0]],
            'Train_End': [train_end_idx],
            'Val_End': [val_end_idx],
            'Total': [len(matrices)]
        })
    elif method == 'scale_only':
        params_df = pd.DataFrame({
            'Method': [method],
            'Vol_Mean': [standardizer.vol_mean],
            'Vol_Std': [standardizer.vol_std],
            'Cov_Mean': [standardizer.cov_scaler.mean_[0]],
            'Cov_Std': [standardizer.cov_scaler.scale_[0]],
            'Train_End': [train_end_idx],
            'Val_End': [val_end_idx],
            'Total': [len(matrices)]
        })
    else:
        # Generic params for other methods
        params_df = pd.DataFrame({
            'Method': [method],
            'Cov_Mean': [standardizer.cov_scaler.mean_[0]],
            'Cov_Std': [standardizer.cov_scaler.scale_[0]],
            'Train_End': [train_end_idx],
            'Val_End': [val_end_idx],
            'Total': [len(matrices)]
        })
    
    params_df.to_csv(mean_std_csv_path, index=False)
    
    print(f"\n✅ Standardization complete for {file_type}")
    print(f"   Output: {output_file_path}")
    print(f"   Parameters: {mean_std_csv_path}")
    
    # Final verification - check standardized diagonal values
    with h5py.File(output_file_path, 'r') as f:
        sample_keys = sorted(f.keys(), key=int)[:5]
        print(f"\nFinal verification of standardized values:")
        
        for key in sample_keys:
            matrix = f[key][:]
            diag = np.diag(matrix)
            off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
            
            print(f"  Matrix {key}:")
            print(f"    Diagonal: mean={diag.mean():.4f}, std={diag.std():.4f}, "
                  f"min={diag.min():.4f}, max={diag.max():.4f}")
            
            if method in ['minmax', 'scale_only']:
                # These methods should preserve positivity
                if np.any(diag < 0):
                    print(f"    ⚠️  WARNING: Found negative values after {method}!")
            else:
                # Log-transform and robust methods can have negative standardized values
                print(f"    Off-diagonal: mean={off_diag.mean():.6f}, std={off_diag.std():.4f}")


def main():
    """
    Main execution - standardize with proper volatility handling
    """
    print("="*80)
    print("30-MINUTE INTRADAY VOLATILITY STANDARDIZATION WITH LOG-TRANSFORM")
    print("="*80)
    
    # Choose standardization method
    # 'log_transform' is recommended for neural networks as it:
    # 1. Handles the multiplicative nature of volatility
    # 2. Makes the distribution more normal
    # 3. Allows the network to learn in log-space
    method = 'log_transform'  # Options: 'log_transform', 'scale_only', 'robust_scale', 'minmax'
    
    print(f"\nUsing standardization method: {method}")
    print("This preserves the positive nature of volatility while normalizing for neural networks")
    
    # Standardize all four SpotV2Net features
    standardize_matrices('vols', method=method, suffix='30min')
    standardize_matrices('covol', method=method, suffix='30min')  # 🔧 ADDED
    standardize_matrices('volvols', method=method, suffix='30min')
    standardize_matrices('covolvols', method=method, suffix='30min')  # 🔧 ADDED
    
    print("\n" + "="*80)
    print("✅ ALL STANDARDIZATION COMPLETE")
    print("="*80)
    print("\nKey features:")
    print("1. All four SpotV2Net features standardized (Vol, CoVol, VolVol, CoVolVol)")
    print("2. Volatilities remain interpretable (log-space for neural networks)")
    print("3. No artificial negative volatilities")
    print("4. Proper train/val/test splits maintained")
    print("5. Covariances properly normalized")
    print("\nThe data is now ready for model training with physically meaningful values!")


if __name__ == "__main__":
    main()