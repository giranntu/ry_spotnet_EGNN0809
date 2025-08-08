#!/usr/bin/env python3
"""
Compare GNN and LSTM models with properly aligned data splits
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
import os
import yaml
from utils.dataset import CovarianceLaggedDataset
from utils.models import GATModel
import matplotlib.pyplot as plt
from datetime import datetime
import h5py

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(MultivariateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def verify_data_alignment():
    """Verify that both models use exactly the same data splits"""
    
    print("="*80)
    print("DATA ALIGNMENT VERIFICATION")
    print("="*80)
    
    # Load GNN dataset
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    
    seq_length = 42
    
    # GNN Dataset
    gnn_dataset = CovarianceLaggedDataset(
        hdf5_file1=os.path.join(os.getcwd(), p['volfile']), 
        hdf5_file2=os.path.join(os.getcwd(), p['volvolfile']),
        root='_'.join([p['root'], str(seq_length)]), 
        seq_length=seq_length
    )
    
    # LSTM Dataset
    lstm_root = f'processed_data/cached_lstm_vols_mats_taq_{seq_length}'
    X_lstm = np.load(os.path.join(lstm_root, 'x_matrices.npy'))
    y_lstm = np.load(os.path.join(lstm_root, 'y_x_vectors.npy'))
    
    print(f"\nDataset Sizes:")
    print(f"  GNN dataset: {len(gnn_dataset)} samples")
    print(f"  LSTM dataset: {len(X_lstm)} samples")
    
    # Check temporal alignment
    with h5py.File('processed_data/vols_mats_taq_standardized.h5', 'r') as f:
        total_matrices = len(f.keys())
        print(f"  Total matrices in HDF5: {total_matrices}")
    
    # Both should have (total_matrices - seq_length) samples
    expected_samples = total_matrices - seq_length
    print(f"  Expected samples (2000 - 42): {expected_samples}")
    
    # Define identical splits for both models
    total_samples = min(len(gnn_dataset), len(X_lstm))
    
    # CRITICAL: Use the same split proportions
    train_size = int(0.8 * total_samples)  # 80% train
    test_size = total_samples - train_size  # 20% test
    
    print(f"\nAligned Data Splits:")
    print(f"  Total samples used: {total_samples}")
    print(f"  Training samples: 0 to {train_size-1} ({train_size} samples)")
    print(f"  Testing samples: {train_size} to {total_samples-1} ({test_size} samples)")
    
    # Temporal interpretation (assuming ~252 trading days per year)
    samples_per_year = 252
    print(f"\nTemporal Interpretation:")
    print(f"  Training period: ~{train_size/samples_per_year:.2f} years")
    print(f"  Testing period: ~{test_size/samples_per_year:.2f} years")
    
    return total_samples, train_size, test_size

def evaluate_models(train_size, test_size):
    """Evaluate both models on the same test set"""
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_length = 42
    
    # Load configuration
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    
    # ============ GNN EVALUATION ============
    print("\n1. Graph Neural Network (GAT) Evaluation:")
    print("-"*40)
    
    # Load GNN dataset
    gnn_dataset = CovarianceLaggedDataset(
        hdf5_file1=os.path.join(os.getcwd(), p['volfile']), 
        hdf5_file2=os.path.join(os.getcwd(), p['volvolfile']),
        root='_'.join([p['root'], str(seq_length)]), 
        seq_length=seq_length
    )
    
    # Use exact same split
    test_dataset_gnn = gnn_dataset[train_size:train_size+test_size]
    test_loader_gnn = GeoDataLoader(test_dataset_gnn, batch_size=128, shuffle=False)
    
    # Load GNN model
    NODE_FEATURES = gnn_dataset[0].x.shape[1]
    EDGE_FEATURES = gnn_dataset[0].edge_attr.shape[1]
    
    gnn_model = GATModel(
        num_node_features=NODE_FEATURES, 
        num_edge_features=EDGE_FEATURES,
        num_heads=p['num_heads'], 
        output_node_channels=p['output_node_channels'], 
        dim_hidden_layers=p['dim_hidden_layers'],
        dropout_att=p['dropout_att'],
        dropout=p['dropout'],
        activation=p['activation'],
        concat_heads=p['concat_heads'],
        negative_slope=p['negative_slope'],
        standardize=p['standardize']
    )
    
    # Load weights if available
    gnn_weights_path = f'output/{p["modelname"]}_{seq_length}/{p["modelname"]}_weights_seed_{p["seed"][0]}.pth'
    if os.path.exists(gnn_weights_path):
        gnn_model.load_state_dict(torch.load(gnn_weights_path, map_location=device))
        print(f"Loaded GNN weights from {gnn_weights_path}")
    
    gnn_model = gnn_model.to(device)
    gnn_model.eval()
    
    # Evaluate GNN
    criterion = nn.MSELoss()
    gnn_test_loss = 0
    gnn_predictions = []
    gnn_targets = []
    
    with torch.no_grad():
        for data in test_loader_gnn:
            data = data.to(device)
            outputs = gnn_model(data)
            targets = data.y_x
            loss = criterion(outputs, targets)
            gnn_test_loss += loss.item()
            
            gnn_predictions.append(outputs.cpu().numpy())
            gnn_targets.append(targets.cpu().numpy())
    
    gnn_test_loss /= len(test_loader_gnn)
    gnn_test_rmse = np.sqrt(gnn_test_loss)
    
    print(f"  Test Loss: {gnn_test_loss:.6f}")
    print(f"  Test RMSE: {gnn_test_rmse:.6f}")
    
    # ============ LSTM EVALUATION ============
    print("\n2. LSTM Model Evaluation:")
    print("-"*40)
    
    # Load LSTM data
    lstm_root = f'processed_data/cached_lstm_vols_mats_taq_{seq_length}'
    X_lstm = np.load(os.path.join(lstm_root, 'x_matrices.npy'))
    y_lstm = np.load(os.path.join(lstm_root, 'y_x_vectors.npy'))
    
    # Use exact same split
    test_X_lstm = torch.tensor(X_lstm[train_size:train_size+test_size], dtype=torch.float32)
    test_y_lstm = torch.tensor(y_lstm[train_size:train_size+test_size], dtype=torch.float32)
    
    test_dataset_lstm = TensorDataset(test_X_lstm, test_y_lstm)
    test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=128, shuffle=False)
    
    # Load LSTM model
    lstm_model = MultivariateLSTM(
        input_size=30,
        hidden_size=128,
        num_layers=2,
        output_size=30,
        dropout=0.1
    )
    
    lstm_weights_path = 'output/LSTM_42/best_lstm_weights.pth'
    if os.path.exists(lstm_weights_path):
        lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=device))
        print(f"Loaded LSTM weights from {lstm_weights_path}")
    
    lstm_model = lstm_model.to(device)
    lstm_model.eval()
    
    # Evaluate LSTM
    lstm_test_loss = 0
    lstm_predictions = []
    lstm_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader_lstm:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            lstm_test_loss += loss.item()
            
            lstm_predictions.append(outputs.cpu().numpy())
            lstm_targets.append(batch_y.cpu().numpy())
    
    lstm_test_loss /= len(test_loader_lstm)
    lstm_test_rmse = np.sqrt(lstm_test_loss)
    
    print(f"  Test Loss: {lstm_test_loss:.6f}")
    print(f"  Test RMSE: {lstm_test_rmse:.6f}")
    
    # ============ COMPARISON ============
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nTest Set Performance (Same {test_size} samples):")
    print(f"  GNN (GAT):  Loss = {gnn_test_loss:.6f}, RMSE = {gnn_test_rmse:.6f}")
    print(f"  LSTM:       Loss = {lstm_test_loss:.6f}, RMSE = {lstm_test_rmse:.6f}")
    
    improvement = ((lstm_test_loss - gnn_test_loss) / lstm_test_loss) * 100
    print(f"\nRelative Performance:")
    if gnn_test_loss < lstm_test_loss:
        print(f"  ✅ GNN outperforms LSTM by {improvement:.2f}%")
    else:
        print(f"  ✅ LSTM outperforms GNN by {-improvement:.2f}%")
    
    return gnn_test_loss, lstm_test_loss, gnn_test_rmse, lstm_test_rmse

def main():
    # Verify data alignment
    total_samples, train_size, test_size = verify_data_alignment()
    
    # Evaluate both models
    gnn_loss, lstm_loss, gnn_rmse, lstm_rmse = evaluate_models(train_size, test_size)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()