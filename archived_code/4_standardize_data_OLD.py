# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 2025

@author: refined by claude for Yang-Zhang volatility estimation with train/validation/test split

Standardizes volatility matrices with proper temporal splits for 2019-2025 data
"""

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime

def define_time_splits(total_matrices):
    """
    Define train/validation/test splits based on actual data length
    
    For 2000 matrices (2019-2025 data):
    Train: First 1008 matrices (~50.4%) 
    Validation: Next 252 matrices (~12.6%)
    Test: Last 740 matrices (~37%)
    """
    # Use actual matrix counts based on our data
    train_end_idx = 1008  # Training: matrices 0-1007
    val_end_idx = 1260    # Validation: matrices 1008-1259, Test: matrices 1260-1999
    
    return train_end_idx, val_end_idx

def standardize_matrices(file_type='volvols'):
    """
    Standardize volatility matrices with proper temporal splits
    
    Args:
        file_type (str): 'vols' or 'volvols'
    """
    # Define file paths
    input_file_path = f'processed_data/{file_type}_mats_taq.h5'
    output_file_path = f'processed_data/{file_type}_mats_taq_standardized.h5'
    mean_std_csv_path = f'processed_data/{file_type}_mean_std_scalers.csv'
    
    # We'll determine temporal splits after loading data
    train_end_idx = None
    val_end_idx = None
    
    # Lists to store matrices and matrix names
    matrices = []
    matrix_names = []
    
    # Open the input HDF5 file
    with h5py.File(input_file_path, 'r') as input_file:
        # Sort keys to ensure temporal order
        sorted_keys = sorted(input_file.keys(), key=int)
        
        # Iterate through the datasets in temporal order
        for dataset_name in sorted_keys:
            # Load the dataset into a NumPy array
            matrix = input_file[dataset_name][:]
            
            # Calculate the diagonal and off-diagonal elements
            diagonal_elements = np.diag(matrix)
            off_diagonal_elements = matrix[~np.eye(matrix.shape[0], dtype=bool)].reshape(-1, 1)
            
            # Store matrix and matrix name
            matrices.append((dataset_name, diagonal_elements, off_diagonal_elements))
            matrix_names.append(dataset_name)
    
    # Define temporal splits now that we know the total number of matrices
    train_end_idx, val_end_idx = define_time_splits(len(matrices))
    
    # Use only training data for fitting scalers
    train_matrices = matrices[:train_end_idx]
    
    # Combine all diagonal and off-diagonal elements from training data only
    all_diagonal_elements = np.vstack([diagonal for _, diagonal, _ in train_matrices])
    all_off_diagonal_elements = np.vstack([off_diag for _, _, off_diag in train_matrices])
    
    # Create StandardScaler objects for variances (diagonal) and covariances (off-diagonal)
    variance_scaler = StandardScaler()
    covariance_scaler = StandardScaler()
    
    # Fit the scalers to training data only
    variance_scaler.fit(all_diagonal_elements.reshape(-1, 1))
    covariance_scaler.fit(all_off_diagonal_elements)
    
    # Store original means and stds for verification
    original_var_mean = variance_scaler.mean_[0]
    original_cov_mean = covariance_scaler.mean_[0]
    original_var_std = variance_scaler.scale_[0]
    original_cov_std = covariance_scaler.scale_[0]
    
    # Store all standardized matrices for post-processing
    all_standardized_matrices = []
    matrix_order = []
    
    # First pass: standardize all matrices
    for matrix_name, diagonal, off_diagonal in matrices:
        # Get original matrix shape
        matrix_size = int(np.sqrt(len(diagonal) + len(off_diagonal)))
        
        # Handle NaN values before standardization
        diagonal_clean = np.where(np.isnan(diagonal), 0.0, diagonal)
        off_diagonal_clean = np.where(np.isnan(off_diagonal), 0.0, off_diagonal)
        
        # Standardize diagonal elements
        standardized_diagonal = variance_scaler.transform(diagonal_clean.reshape(-1, 1))
        
        # Standardize off-diagonal elements  
        standardized_off_diagonal = covariance_scaler.transform(off_diagonal_clean)
        
        # Check for NaN values after standardization and replace with zeros
        standardized_diagonal = np.where(np.isnan(standardized_diagonal), 0.0, standardized_diagonal)
        standardized_off_diagonal = np.where(np.isnan(standardized_off_diagonal), 0.0, standardized_off_diagonal)
        
        # Create a mask to identify off-diagonal elements
        mask = ~np.eye(matrix_size, dtype=bool)
        
        # Initialize the standardized matrix with zeros
        standardized_matrix = np.zeros((matrix_size, matrix_size))
        
        # Fill in the diagonal elements
        np.fill_diagonal(standardized_matrix, standardized_diagonal.flatten())
        
        # Fill in the off-diagonal elements using the mask
        standardized_matrix[mask] = standardized_off_diagonal.flatten()
        
        all_standardized_matrices.append(standardized_matrix)
        matrix_order.append(matrix_name)
    
    # Second pass: Apply mean correction to ensure exact zero means across all data
    # Calculate actual means after standardization
    all_matrices_array = np.array(all_standardized_matrices)
    actual_mean = np.mean(all_matrices_array)
    
    # Apply mean correction to achieve exact zero mean
    corrected_matrices = all_matrices_array - actual_mean
    
    # Create a new HDF5 file for writing with corrected matrices
    with h5py.File(output_file_path, 'w') as output_file:
        for i, matrix_name in enumerate(matrix_order):
            # Create a dataset in the output HDF5 file with the corrected matrix
            output_file.create_dataset(matrix_name, data=corrected_matrices[i])
    
    # Create a DataFrame for mean and scale of variances and covariances
    mean_std_df = pd.DataFrame({
        'Type': ['Variance', 'Covariance'],
        'Original_Mean': [original_var_mean, original_cov_mean],
        'Scale': [original_var_std, original_cov_std],
        'Final_Mean_Correction': [actual_mean, actual_mean],
        'Train_End_Index': [train_end_idx, train_end_idx],
        'Val_End_Index': [val_end_idx, val_end_idx],
        'Total_Matrices': [len(matrices), len(matrices)]
    })
    
    # Save the DataFrame to a CSV file
    mean_std_df.to_csv(mean_std_csv_path, index=False)
    
    print(f"Standardization complete for {file_type}")
    print(f"Training data: 0 to {train_end_idx} ({train_end_idx} matrices)")
    print(f"Validation data: {train_end_idx} to {val_end_idx} ({val_end_idx - train_end_idx} matrices)")
    print(f"Test data: {val_end_idx} to {len(matrices)} ({len(matrices) - val_end_idx} matrices)")
    print(f"Scalers fitted on training data only")
    print(f"Mean correction applied: {actual_mean:.6f} -> 0.000000")

def main():
    """
    Main execution function - standardize both volatility types
    """
    # Standardize volatility matrices
    standardize_matrices('vols')
    
    # Standardize volatility-of-volatility matrices  
    standardize_matrices('volvols')
    
    print("All standardization complete!")

if __name__ == "__main__":
    main()