# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:54:34 2023

@author: ab978
"""
import pandas as pd
import numpy as np
import os, pdb
from tqdm import tqdm
from glob import glob
import h5py
from collections import OrderedDict

def generate_matrices(source='price', suffix='30min'):
    
    ################################### PRICES################################################
    if source == 'price':
        filenames = ['covol', 'vol']
        # filenames = ['covol202306_09', 'vol202306_09']
    elif source == 'vol':
        filenames = ['covol_of_vol', 'vol_of_vol']
        
    # get historical cov series saved as files with TICKER1_TICKER2 filename
    symbols = []
    hist_covs = {}
    for filename in tqdm(glob(os.path.join(os.getcwd(),'processed_data',filenames[0],'*.csv')),desc='Getting symbols...'):
        df = pd.read_csv(filename, header = None)
        # df = df.stack().reset_index(drop=True) FIXED MISTAKE
        df = pd.concat([df[col] for col in df], ignore_index=True)
        pairs = os.path.basename(filename).split('.csv')[0]
        symbols.extend(pairs.split('_'))
        hist_covs[pairs] = df.values
        

    # get historical vol series saved as TICKER filename  
    df_to_concat = []
    for filename in tqdm(glob(os.path.join(os.getcwd(),'processed_data',filenames[1],'*.csv')),desc='Joining RVols...'):
        df = pd.read_csv(filename, header=None)
        # df = df.stack().reset_index(drop=True) FIXED MISTAKE
        df = pd.concat([df[col] for col in df], ignore_index=True)
        symbol = os.path.basename(filename).split('.')[0]
        df.name = symbol
        df_to_concat.append(df)
        
    vol_df = pd.concat(df_to_concat,axis=1)
    sorted_columns = sorted(vol_df.columns)
    vol_df = vol_df[sorted_columns]
    

    # check symbols between covs and vols files
    ordered_symbols = sorted(list(set(symbols)))
    
    assert len(ordered_symbols) == (ordered_symbols == vol_df.columns).sum()
    # get the number of periods (covariance matrices)
    n_periods = list(set([len(hist_covs[k]) for k in hist_covs.keys()]))[0]
    print('Vol has {} obs'.format(n_periods))
    
    # initialize dict of empty pandas dataframe
    hist_cov_mat = {}
    # hist_cov_mat = OrderedDict()
    for i in tqdm(iterable=range(n_periods), desc='Creating empty cov mats'):
        hist_cov_mat[str(i)] = pd.DataFrame(index=ordered_symbols, columns = ordered_symbols)
    

    # insert diagonals (variances) into the dataframes
    for i in tqdm(iterable=vol_df.index, desc='Filling Mat with Vols'):
        np.fill_diagonal(hist_cov_mat[str(i)].values,vol_df.iloc[i].values)



    # insert all the other values
    for k in tqdm(iterable=hist_covs.keys(), desc='Filling Mat with Covs...'):
        s1,s2 = k.split('_')
        values = hist_covs[str(k)].reshape(-1)
        for i,v in enumerate(values):
            hist_cov_mat[str(i)].loc[s1,s2] = v
            hist_cov_mat[str(i)].loc[s2,s1] = v
   
    # save df as numpy values
    hist_cov_mat_numpy = {k:v.values.astype(np.float64) for k,v in hist_cov_mat.items()}
    
    return hist_cov_mat_numpy
        
def are_keys_ordered_as_numbers(dictionary):
    keys = list(dictionary.keys())
    for i in range(len(keys) - 1):
        if int(keys[i]) > int(keys[i + 1]):
            return False
    return True

if __name__=='__main__':
    
    # Check if 30-minute data exists (from script 2)
    import os
    if os.path.exists("processed_data/vols_mats_30min.h5"):
        print("✅ 30-minute HDF5 files already exist from script 2")
        print("   Files: vols_mats_30min.h5 and volvols_mats_30min.h5")
        print("   These were created directly by 2_organize_prices_as_tables.py")
        print("   Script 3 is not needed for 30-minute data processing")
    else:
        # Legacy daily data processing (if needed)
        suffix = 'taq'  # For daily data
        
        vol = generate_matrices(source='price', suffix=suffix)
        assert all(int(list(vol.keys())[i]) <= int(list(vol.keys())[i + 1]) for i in range(len(list(vol.keys())) - 1))

        volvol = generate_matrices(source='vol', suffix=suffix)
        assert all(int(list(volvol.keys())[i]) <= int(list(volvol.keys())[i + 1]) for i in range(len(list(volvol.keys())) - 1))
        
        # align observation
        vol = {k: v for k, v in sorted(vol.items(), key=lambda x: int(x[0]))[:len(volvol)]}

        # Save the covariance matrices in an HDF5 file
        with h5py.File(f"processed_data/vols_mats_{suffix}.h5", "w") as f:
            for key, value in vol.items():
                # Create a dataset with the same name as the key and store the value
                f.create_dataset(str(key), data=value, dtype=np.float64)
        # Save the covariance matrices in an HDF5 file
        with h5py.File(f"processed_data/volvols_mats_{suffix}.h5", "w") as f:
            for key, value in volvol.items():
                # Create a dataset with the same name as the key and store the value
                f.create_dataset(str(key), data=value, dtype=np.float64)

# Code left here for further check
# # Load back the data
# # Open the HDF5 file for reading
# with h5py.File("processed_data/covs_mats_30min2.h5", 'r') as f:
#     # Create an empty dictionary to store the loaded data
#     data_dict_loaded = {}
    
#     # Loop through each dataset in the file and add it to the dictionary
#     for key in f.keys():
#         data_dict_loaded[int(key)] = np.array(f[key])
        
        
# # Assume data_dict is the dictionary containing the arrays
# is_symmetric = {}

# # Loop through each array in the dictionary
# for key, arr in hist_cov_mat_numpy.items():
#     # Check if the array is diagonal by comparing it with its diagonal elements
#     if not np.allclose(arr, arr.T):
#         print(key, 'Not symm')