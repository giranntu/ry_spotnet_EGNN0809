#!/usr/bin/env python3
"""
LSTM Training Script with Proper Temporal Splits
================================================
This script trains LSTM model with exact same temporal splits as GNN
to ensure fair comparison and research rigor.

Aligned with Script 4 (standardization) splits:
- Train: matrices 0-1008 
- Val: matrices 1008-1260
- Test: matrices 1260-2000
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import yaml
import math
import h5py

class LSTMModel(nn.Module):
    """Simple LSTM for volatility prediction"""
    def __init__(self, input_size=900, hidden_size=256, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 900)  # Predict next matrix (30x30 flattened)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        out = self.fc(lstm_out[:, -1, :])
        return out

def prepare_lstm_data(seq_length=42):
    """Load and prepare LSTM data with proper temporal alignment"""
    
    print("Loading volatility data...")
    vols_file = 'processed_data/vols_mats_taq_standardized.h5'
    
    sequences = []
    targets = []
    
    with h5py.File(vols_file, 'r') as f:
        # Sort keys to ensure temporal order
        sorted_keys = sorted(f.keys(), key=int)
        matrices = [f[key][:].flatten() for key in sorted_keys]
    
    # Create sequences
    for i in range(len(matrices) - seq_length):
        seq = np.array(matrices[i:i+seq_length])
        target = matrices[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    
    print(f"Created {len(X)} sequences with seq_length={seq_length}")
    
    # Define splits aligned with Script 4
    # Matrix indices from script 4: train=0-1008, val=1008-1260, test=1260-2000
    # Adjust for sequence length offset
    seq_offset = seq_length - 1
    train_end_idx = max(0, 1008 - seq_offset)  # 967 for seq_length=42
    val_end_idx = max(0, 1260 - seq_offset)    # 1219 for seq_length=42
    
    # Split data
    X_train = X[:train_end_idx]
    y_train = y[:train_end_idx]
    X_val = X[train_end_idx:val_end_idx]
    y_val = y[train_end_idx:val_end_idx]
    X_test = X[val_end_idx:]
    y_test = y[val_end_idx:]
    
    print(f"\nTemporal splits (aligned with Script 4):")
    print(f"  Train: 0-{train_end_idx} ({len(X_train)} samples)")
    print(f"  Val: {train_end_idx}-{val_end_idx} ({len(X_val)} samples)")
    print(f"  Test: {val_end_idx}-{len(X)} ({len(X_test)} samples)")
    print(f"  Total: {len(X)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_lstm(seed=42):
    """Train LSTM with proper temporal splits"""
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Load hyperparameters from GNN config for consistency
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_lstm_data(p['seq_length'])
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    # Create dataloaders
    batch_size = p['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cpu")  # Match GNN training
    model = LSTMModel(input_size=900, hidden_size=256, num_layers=2, dropout=0.2).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    
    # Training setup
    num_epochs = p['num_epochs']
    tolerance = p['tolerance']
    
    # Create output folder
    folder_path = f'output/LSTM_{p["seq_length"]}_aligned'
    os.makedirs(folder_path, exist_ok=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("\n" + "="*80)
    print("Starting LSTM Training with Aligned Temporal Splits")
    print("="*80)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train'):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Val'):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model
        tolerance_val = float(p.get('tolerance', 1e-9))  # Convert to float
        if epoch == 0 or avg_val_loss + tolerance_val < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_path = f'{folder_path}/lstm_weights_seed_{seed}.pth'
            torch.save(model.state_dict(), save_path)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.10f}, Val Loss={avg_val_loss:.10f}, Best Epoch={best_epoch+1}")
    
    # Final test evaluation
    print("\n" + "="*80)
    print(f"Final evaluation on test set with best model from epoch {best_epoch+1}")
    print("="*80)
    
    # Load best model (weights_only=False for compatibility)
    model.load_state_dict(torch.load(save_path, weights_only=False))
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc='Final test evaluation'):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
    
    final_test_loss = test_loss / len(test_loader)
    final_test_rmse = math.sqrt(final_test_loss)
    
    print(f"Final Test Loss: {final_test_loss:.10f}, Test RMSE: {final_test_rmse:.10f}")
    
    # Save losses
    np.save(f'{folder_path}/train_losses_seed_{seed}.npy', np.array(train_losses))
    np.save(f'{folder_path}/val_losses_seed_{seed}.npy', np.array(val_losses))
    np.save(f'{folder_path}/test_loss_seed_{seed}.npy', np.array([final_test_loss]))
    
    # Save config for reproducibility
    config = {
        'model': 'LSTM',
        'seed': seed,
        'seq_length': p['seq_length'],
        'batch_size': batch_size,
        'learning_rate': p['learning_rate'],
        'num_epochs': num_epochs,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'best_epoch': best_epoch + 1,
        'final_test_loss': final_test_loss,
        'final_test_rmse': final_test_rmse
    }
    
    with open(f'{folder_path}/config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nTraining complete! Results saved to {folder_path}")
    return final_test_loss

if __name__ == "__main__":
    # Train with same seed as GNN for fair comparison
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    
    seed = p['seed'][0] if isinstance(p['seed'], list) else p['seed']
    train_lstm(seed=seed)