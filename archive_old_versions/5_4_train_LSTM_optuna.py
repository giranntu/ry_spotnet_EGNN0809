# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 09:17:48 2023

@author: ab978
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import optuna
import os

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(MultivariateLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def objective(trial):
    
    folder_path_trial = 'output/{}_{}/{}'.format('20231010_lstm_tuning','42',trial.number)
    if not os.path.exists(folder_path_trial):
        os.makedirs(folder_path_trial)
    
    # Hyperparameters to be optimized
    hidden_size = trial.suggest_int("hidden_size", 64, 128)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    
    root = '_'.join(['processed_data/cached_lstm_vols_mats_taq','42'])
    # Load data
    X = np.load('/'.join([root,'x_matrices.npy']))
    y = np.load('/'.join([root,'y_x_vectors.npy']))
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Create a TensorDataset and DataLoader
    train_dataset = TensorDataset(X_tensor, y_tensor)
    
    # Perform train-test split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = train_dataset[0][0].shape[1]
    output_size = train_dataset[0][1].shape[0]
    model = MultivariateLSTM(input_size, hidden_size, num_layers, output_size, dropout)

    
    criterion = nn.MSELoss()

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:  # 'adamw'
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer your model to the device (this will be GPU if available, else CPU)
    model = model.to(device)

    best_val_loss = float("inf")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device) * 10000
            targets = targets.to(device) * 10000

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device) * 10000
                targets = targets.to(device) * 10000

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(folder_path_trial, 'best_lstm_weights.pth'))
            best_val_loss = avg_val_loss
            
    
        # Compute RMSE
        avg_train_rmse = np.sqrt(avg_train_loss)
        avg_val_rmse = np.sqrt(avg_val_loss)
        
        # Print the losses and RMSE
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.10f}, Train RMSE: {avg_train_rmse:.10f}, Validation Loss: {avg_val_loss:.10f}, Validation RMSE: {avg_val_rmse:.10f}")
    

    return best_val_loss  # Objective is to minimize validation loss

if __name__ == "__main__":
    
    set_seed()
    
    folder_path = 'output/{}_{}'.format('20231010_lstm_tuning','42')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
        
    study.trials_dataframe().to_csv('{}/study.csv'.format(folder_path))
