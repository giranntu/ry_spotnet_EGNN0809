# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:37:17 2023

@author: ab978
"""
import numpy as np
import pdb, os
import yaml
import optuna
import sys
from train_RGNN import train



def objective(trial):
    # Load hyperparam file
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    
    # Access the hyperparameters and grid from the config
    hyperparameters = p['hyperparameters']
    grid = p['grid']
    # Iterate over the hyperparameters and update p accordingly
    for param, (values,dtype) in hyperparameters.items():
        if param in grid:
            if dtype == 'cat':
                p[param] = trial.suggest_categorical(param, values)
            elif dtype == 'int':
                if len(values) == 3:
                    p[param] = trial.suggest_int(param, values[0], values[1], step=values[2])
                else:
                    p[param] = trial.suggest_int(param, values[0], values[1])
            elif dtype == 'float':
                if len(values) == 3:
                    p[param] = trial.suggest_float(param, values[0], values[1], step=values[2])
                else:
                    p[param] = trial.suggest_float(param, values[0], values[1])
            else:
                print('Choose an available dtype!')
                sys.exit()

    # Run the training process with the trial-specific hyperparameters
    train(trial=trial,p=p)
    
    # Load the test loss from the saved file
    folder_path = 'output/{}_optuna/{}'.format(p['modelname'], trial.number)
    test_losses = np.load('{}/test_losses_seed_{}.npy'.format(folder_path, p['seed']))
    
    # Return the minimum test loss as the optimization objective
    return np.min(test_losses)


if __name__ == '__main__':
    
    # Load hyperparam file
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    
    # Optimize the hyperparameters using Optuna
    if p['random_sampler']:
        study = optuna.create_study(study_name=p['modelname'],direction='minimize', sampler=optuna.samplers.RandomSampler())
    else :
        study = optuna.create_study(study_name=p['modelname'],direction='minimize')
    
    study.optimize(objective, n_trials=p['n_trials'])  # Adjust the number of trials as needed
    
    # Print the best hyperparameters and the corresponding test loss
    print('Best trial:')
    print('  Value: ', study.best_trial.value)
    print('  Params: ')
    for key, value in study.best_trial.params.items():
        print('{}: {}'.format(key, value))
            
    # Store the study
    study.trials_dataframe().to_csv('output/{}_optuna/study.csv'.format(p['modelname']))