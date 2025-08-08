# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:37:17 2023

@author: ab978
"""

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import pdb, os
import yaml
import math
import sys
import torch.multiprocessing as mp
from utils.models import GATModel, RecurrentGCN
from utils.dataset import (CovarianceLaggedDataset,
                            CovarianceSparseDataset,
                            CovarianceLaggedMultiOutputDataset)
from typing import Optional, Dict, Union

def train(seed: Optional[int] = None,
          trial: Optional[object] = None,
          p: Optional[Dict[str, Union[str, int, float, bool]]] = None) -> None:
    """
    Trains a model based on the given parameters.

    :param seed: Random seed for reproducibility, defaults to None
    :param trial: Optuna trial object for hyperparameter optimization, defaults to None
    :param p: Dictionary containing hyperparameters and other configuration details, defaults to None
    """
    
    if trial and p:
        # Define the folder path
        folder_path = 'output/{}_{}/{}'.format(p['modelname'], 'optuna', trial.number)
    else:
        # Load hyperparam file
        with open('config/GNN_param.yaml', 'r') as f:
            p = yaml.safe_load(f)
        p['seed'] = seed
        # Define the folder path
        folder_path = 'output/{}_{}'.format(p['modelname'],p['seq_length'])
        
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Save the yaml file to the model folder
    with open('{}/GNN_param.yaml'.format(folder_path), 'w') as f:
        yaml.dump(p, f)

    # fix randomness
    torch.manual_seed(p['seed'])
    np.random.seed(p['seed'])
    torch.cuda.manual_seed_all(p['seed'])

    # Instantiate the dataset
    if p['fully_connected']:
        if p['output_node_channels'] == 1:
            dataset = CovarianceLaggedDataset(hdf5_file1=os.path.join(os.getcwd(),p['volfile']), 
                                              hdf5_file2=os.path.join(os.getcwd(),p['volvolfile']),
                                              root='_'.join([p['root'],str(p['seq_length'])]), 
                                              seq_length=p['seq_length'])
        else:
            dataset = CovarianceLaggedMultiOutputDataset(hdf5_file1=os.path.join(os.getcwd(),p['volfile']),
                                                         hdf5_file2=os.path.join(os.getcwd(),p['volvolfile']),
                                                         root='_'.join([p['root'],str(p['seq_length']),'moutput']), 
                                                         seq_length=p['seq_length'], future_steps=p['output_node_channels'])
    else:
        if p['threshold']:
            root = '_'.join([p['root'],'sparse','t_{}'.format(p['threshold']),str(p['seq_length'])])
        else:
            root = '_'.join([p['root'],'sparse',str(p['seq_length'])])
        dataset = CovarianceSparseDataset(hdf5_file=p['datafile'],root=root, seq_length=p['seq_length'], threshold=p['threshold'])
        p['num_edge_features'] = 1
    # pdb.set_trace()  # Commented out for automated execution
    
    # Use proper temporal splits aligned with standardization (script 4)
    # For 2000 matrices with seq_length=42: 1958 valid samples
    # Following the splits from script 4 (standardization):
    # Train: 0-1008 matrices -> samples 0-966 (after seq_length adjustment)
    # Val: 1008-1260 matrices -> samples 966-1218 
    # Test: 1260-2000 matrices -> samples 1218-1958
    
    # Since dataset has seq_length offset, adjust indices accordingly
    seq_offset = p['seq_length'] - 1  # 41 for seq_length=42
    
    # Define split indices based on matrix indices from script 4
    train_end_matrix = 1008
    val_end_matrix = 1260
    
    # Adjust for sequence length to get sample indices
    train_end_idx = max(0, train_end_matrix - seq_offset)  # 967
    val_end_idx = max(0, val_end_matrix - seq_offset)      # 1219
    
    # Create train/val/test datasets with proper temporal splits
    train_dataset = dataset[:train_end_idx]
    val_dataset = dataset[train_end_idx:val_end_idx]
    test_dataset = dataset[val_end_idx:]
    
    print(f"Dataset splits (aligned with script 4 standardization):")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train: 0-{train_end_idx} ({len(train_dataset)} samples)")
    print(f"  Val: {train_end_idx}-{val_end_idx} ({len(val_dataset)} samples)")
    print(f"  Test: {val_end_idx}-{len(dataset)} ({len(test_dataset)} samples)")
    
    # Create DataLoaders for train, val and test datasets
    train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=p['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)
    
    
    # select dimensions from data
    NODE_FEATURES = dataset[0].x.shape[1]
    EDGE_FEATURES = dataset[0].edge_attr.shape[1]

    # Instantiate the model
    if p['modeltype'] == 'gat':
        model = GATModel(num_node_features=NODE_FEATURES, 
                         num_edge_features = EDGE_FEATURES,
                         num_heads=p['num_heads'], 
                         output_node_channels=p['output_node_channels'], 
                         dim_hidden_layers=p['dim_hidden_layers'],
                         dropout_att = p['dropout_att'],
                         dropout = p['dropout'],
                         activation = p['activation'],
                         concat_heads= p['concat_heads'],
                         negative_slope=p['negative_slope'],
                         standardize = p['standardize'])
    elif p['modeltype'] == 'rnn':
        model = RecurrentGCN(num_features=p['seq_length'], 
                         hidden_channels=p['hidden_channels'], 
                         output_node_channels=p['output_node_channels'], 
                         dropout = p['dropout'],
                         activation = p['activation'])
    
        
    # Force CPU usage for debugging NaN issues
    device = torch.device("cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    
  
    # Set loss function and optimizer
    criterion = torch.nn.MSELoss()
    if p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'])
    elif p['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=p['learning_rate'])
    else:
        print('Choose an available optimizer')
        sys.exit()
    
    # Train the model
    train_losses, val_losses, test_losses = [], [], []
    prev_val_loss = float('inf')
    best_epoch = 0
    for epoch in range(p['num_epochs']):
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(tqdm(iterable=train_loader, desc='Training batches...')):
            data = data.to(device)
            if p['scale_up']:
                data.x = data.x * p['scale_up'] 
                data.edge_attr = data.edge_attr * p['scale_up'] 
                data.y_x = data.y_x * p['scale_up']

            # Forward pass
            y_x_hat = model(data)
            # Compute loss
            y_x = data.y_x

            loss = criterion(y_x_hat, y_x)
            
            # Debug: Check for NaN before backward pass
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}")
                print(f"  y_x_hat stats: min={y_x_hat.min():.6f}, max={y_x_hat.max():.6f}, NaN count={torch.isnan(y_x_hat).sum()}")
                print(f"  y_x stats: min={y_x.min():.6f}, max={y_x.max():.6f}, NaN count={torch.isnan(y_x).sum()}")
                break
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients for NaN
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"NaN gradients detected at batch {batch_idx}")
                break
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimize
            optimizer.step()
            total_loss += loss.item()

        # Compute average training loss
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_rmse = math.sqrt(avg_train_loss)

        # Evaluate on the VALIDATION set (best practice: use val for model selection)
        model.eval()
        val_loss = 0
    
        with torch.no_grad():
            for data in tqdm(iterable=val_loader, desc='Validation batches...'):
                data = data.to(device)
                if p['scale_up']:
                    data.x = data.x * p['scale_up'] 
                    data.edge_attr = data.edge_attr * p['scale_up'] 
                    data.y_x = data.y_x * p['scale_up']
                # Forward pass
                y_x_hat = model(data)
                # Compute loss
                y_x = data.y_x
                loss = criterion(y_x_hat, y_x)
                val_loss += loss.item()
        # Compute average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_val_rmse = math.sqrt(avg_val_loss)
        
        # Check if the validation loss improved (best practice: early stopping on val)
        if epoch == 0 or avg_val_loss + float(p['tolerance']) < prev_val_loss:
            # Update the previous validation loss
            prev_val_loss = avg_val_loss
            best_epoch = epoch
            # Save the model weights
            save_path = '{}/{}_weights_seed_{}.pth'.format(folder_path,p['modelname'],p['seed'])
            torch.save(model.state_dict(), save_path)
        
        print(f"Epoch: {epoch+1}/{p['num_epochs']}, Train Loss: {avg_train_loss:.10f}, Val Loss: {avg_val_loss:.10f}, Train RMSE: {avg_train_rmse:.10f}, Val RMSE: {avg_val_rmse:.10f}, Best Epoch: {best_epoch+1}")
   
    # After training, evaluate on the TEST set with best model
    print("\n" + "="*80)
    print("Final evaluation on test set with best model from epoch {}".format(best_epoch+1))
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        for data in tqdm(iterable=test_loader, desc='Final test evaluation...'):
            data = data.to(device)
            if p['scale_up']:
                data.x = data.x * p['scale_up'] 
                data.edge_attr = data.edge_attr * p['scale_up'] 
                data.y_x = data.y_x * p['scale_up']
            # Forward pass
            y_x_hat = model(data)
            # Compute loss
            y_x = data.y_x
            loss = criterion(y_x_hat, y_x)
            test_loss += loss.item()
    
    # Compute final test metrics
    final_test_loss = test_loss / len(test_loader)
    final_test_rmse = math.sqrt(final_test_loss)
    
    print(f"Final Test Loss: {final_test_loss:.10f}, Test RMSE: {final_test_rmse:.10f}")
    
    # save losses
    np.save('{}/train_losses_seed_{}.npy'.format(folder_path, p['seed']), np.array(train_losses))
    np.save('{}/val_losses_seed_{}.npy'.format(folder_path, p['seed']), np.array(val_losses))
    np.save('{}/test_loss_seed_{}.npy'.format(folder_path, p['seed']), np.array([final_test_loss]))


if __name__ == '__main__':
    
    # Load hyperparam file
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    # Set the desired seed(s)
    seeds = p['seed']  # Example seeds

    if len(seeds) > 1:
        mp.set_start_method('spawn')
        # Run the `train` function in parallel with different seeds
        with mp.Pool() as pool:
            pool.map(train, seeds)
    else:
        # Run the `train` function once with a single seed that is specified in the config file
        train(seed=seeds[0])
            
