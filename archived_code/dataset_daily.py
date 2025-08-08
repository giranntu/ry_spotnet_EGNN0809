# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:53:35 2023

@author: ab978
"""

from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import os
import torch
import h5py, pdb
from natsort import natsorted
from tqdm import tqdm

# reshape vs transpose https://discuss.pytorch.org/t/different-between-permute-transpose-view-which-should-i-use/32916

class CovarianceTemporalDataset(InMemoryDataset):
    '''This is the first dataset I implemented where the structure of the data seems not to be the one that the 
    GATConv layer is expected. '''
    def __init__(self, hdf5_file, root='processed_data/cached_datasets_temporal/', transform=None, pre_transform=None, seq_length=None):
        self.hdf5_file = hdf5_file
        self.root = root
        self.seq_length = seq_length
        super(CovarianceTemporalDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file)]

    @property
    def processed_file_names(self):
        return ['data_temp.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        with h5py.File(self.hdf5_file, 'r') as f:
            keys = list(f.keys())
            for i in range(len(f) - self.seq_length):
                seq_data_list = []
                for j in range(self.seq_length):
                    cov_matrix = np.array(f[keys[i+j]])
                    next_cov_matrix  = np.array(f[keys[i+j+1]])
                    
                    # Create the adjacency matrix from the covariance matrix
                    adj_matrix = cov_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the adjacency matrix (excluding diagonal)
                    mask = np.triu(np.ones_like(adj_matrix), k=1) > 0
                    edge_weights = torch.tensor(adj_matrix[mask], dtype=torch.float)
    
                    # Create edge_index tensor
                    edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
    
                    # Extract the variances (diagonal) as node features
                    node_features = np.diag(cov_matrix)
                    x = torch.tensor(node_features, dtype=torch.float).view(-1, 1)
    
                    # Process the next covariance matrix to create target values
                    adj_matrix_next = next_cov_matrix.copy()
                    np.fill_diagonal(adj_matrix_next, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the next adjacency matrix (excluding diagonal)
                    mask = np.triu(np.ones_like(adj_matrix_next), k=1) > 0
                    y_edge_weight = torch.tensor(adj_matrix_next[mask], dtype=torch.float).view(-1, 1)
    
                    # Extract the variances (diagonal) of the next covariance matrix as target node features
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float).view(-1, 1)
                    

                    # Create PyTorch Geometric Data object
                    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights,
                                y_edge_weight=y_edge_weight, y_x=y_x)
                    
                    seq_data_list.append(data)

                    

                # Combine the sequence of Data objects into a single object with the desired format
                # seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=0),
                #                 edge_index=torch.stack([d.edge_index for d in seq_data_list], dim=0),
                #                 edge_weight=torch.stack([d.edge_weight for d in seq_data_list], dim=0),
                #                 y_edge_weight=torch.stack([d.y_edge_weight for d in seq_data_list], dim=0),
                #                 y_x=torch.stack([d.y_x for d in seq_data_list], dim=0))

                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=0).transpose(0,1),
                                edge_index=torch.stack([d.edge_index for d in seq_data_list], dim=0).transpose(0,1),
                                edge_weight=torch.stack([d.edge_weight for d in seq_data_list], dim=0).transpose(0,1),
                                y_edge_weight=torch.stack([d.y_edge_weight for d in seq_data_list], dim=0).transpose(0,1),
                                y_x=torch.stack([d.y_x for d in seq_data_list], dim=0).transpose(0,1))
                
                

                data_list.append(seq_data)

        
        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(os.getcwd(),self.processed_paths[0]))
        
class CovarianceLSTMDataset():
    def __init__(self, hdf5_file1, hdf5_file2, root='processed_data/cached_datasets_lagged/', seq_length=None):
        self.hdf5_file1 = hdf5_file1
        self.hdf5_file2 = hdf5_file2
        self.root = root
        self.seq_length = seq_length

    def process(self):
        x_matrices = []
        y_x_vectors = []
        with h5py.File(self.hdf5_file1, 'r') as f1, h5py.File(self.hdf5_file2, 'r') as f2:
            keys = list(f1.keys())
            keys = natsorted(keys)
            for i in tqdm(iterable=range(len(f1) - self.seq_length),desc='Creating LSTM dataset...'):
                seq_data_list = []
                for j in range(self.seq_length):
                    cov_matrix = np.array(f1[keys[i+j]])
                    covol_matrix = np.array(f2[keys[i+j]])

                    # Extracting vols (diagonal of cov_matrix)
                    vols = np.diag(cov_matrix)
                    # Extracting covols (unique values outside the diagonals of cov_matrix)
                    covols = cov_matrix[np.triu_indices(cov_matrix.shape[0], k=1)]
                    # Extracting volvol (diagonal of covol_matrix)
                    volvol = np.diag(covol_matrix)
                    # Extracting covolvols (unique values outside the diagonals in covol_matrix)
                    covolvols = covol_matrix[np.triu_indices(covol_matrix.shape[0], k=1)]

                    # Concatenating all the features
                    features = np.concatenate([vols, covols, volvol, covolvols])
                    x = torch.tensor(features, dtype=torch.float)

                    # Assuming the target is the diagonal of the next covariance matrix
                    next_cov_matrix = np.array(f1[keys[i+j+1]])
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float)

                    seq_data_list.append((x, y_x))

                x_matrix = torch.stack([data[0] for data in seq_data_list], dim=0).numpy()
                y_x_vector = seq_data_list[-1][-1].numpy()
                x_matrices.append(x_matrix)
                y_x_vectors.append(y_x_vector)

        # Ensure the directory exists
        os.makedirs(self.root, exist_ok=True)
    
        # Saving the entire dataset as .npy files in the specified directory
        # Use proper temporal split instead of hardcoded cutoff
        train_start_idx = max(0, len(x_matrices) - int(len(x_matrices) * 0.8))
        np.save(os.path.join(self.root, 'x_matrices.npy'), x_matrices[train_start_idx:])
        np.save(os.path.join(self.root, 'y_x_vectors.npy'), y_x_vectors[train_start_idx:])
       
        
class CovarianceLaggedDataset(InMemoryDataset):
    def __init__(self, hdf5_file1, hdf5_file2, root='processed_data/cached_datasets_lagged/', transform=None, pre_transform=None, seq_length=None):
        self.hdf5_file1 = hdf5_file1
        self.hdf5_file2 = hdf5_file2
        self.root = root
        self.seq_length = seq_length
        super(CovarianceLaggedDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file1), os.path.basename(self.hdf5_file2)]


    @property
    def processed_file_names(self):
        return ['data_temp.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        # Open both HDF5 files simultaneously
        with h5py.File(self.hdf5_file1, 'r') as f1, h5py.File(self.hdf5_file2, 'r') as f2:
            keys = list(f1.keys())
            # order keys
            keys = natsorted(keys)
            ordered = all(int(keys[i]) <= int(keys[i+1]) for i in range(len(keys)-1))
            assert ordered, 'Keys of files 1 have not been ordered'
            keys = list(f2.keys())
            # order keys
            keys = natsorted(keys)
            ordered = all(int(keys[i]) <= int(keys[i+1]) for i in range(len(keys)-1))
            assert ordered, 'Keys of files 2 have not been ordered'
            # # TODO cutoff hardcoded
            # for i in range(len(f)-int(len(f)*0.2), len(f) - self.seq_length):
            for i in range(len(f1) - self.seq_length):
                seq_data_list = []
                for j in range(self.seq_length):
                    # vol data
                    cov_matrix = np.array(f1[keys[i+j]])
                    next_cov_matrix  = np.array(f1[keys[i+j+1]])
                    # volvol data
                    covol_matrix = np.array(f2[keys[i+j]])
                    next_covol_matrix  = np.array(f2[keys[i+j+1]])
                    assert int(keys[i+j]) + 1 == int(keys[i+j+1]), 'The labeling process is not considering consecutive matrices in file2'

                    # Create the adjacency matrix from the covariance matrix
                    adj_matrix = covol_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the adjacency matrix (excluding diagonal)
                    mask = np.triu(np.ones_like(adj_matrix), k=1) > 0

                    # Create edge_index tensor
                    # edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()  
                    
                    # Create edge_index tensor for upper triangle
                    edge_index_upper = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
                    # Create edge_index tensor for lower triangle (transpose of upper)
                    edge_index_lower = edge_index_upper[[1, 0], :]
                    # Combine both to get full edge_index
                    edge_index = torch.cat([edge_index_upper, edge_index_lower], dim=1)

                    # edge_attr = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    # Extract the variances (diagonal) as a separate tensor
                    variances = torch.tensor(np.diag(covol_matrix), dtype=torch.float)
                    # Get the source and target indices from the edge_index
                    source_indices = edge_index[0]
                    target_indices = edge_index[1]
                    # Extract the variances for the source and target nodes
                    source_variances = variances[source_indices]
                    target_variances = variances[target_indices]
                    # Create the original edge attributes (covariances)
                    covariances = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    # Duplicate the covariances for the lower triangle
                    covariances = torch.cat([covariances, covariances])
                    # Concatenate the covariances with the source and target variances
                    edge_attr = torch.stack([covariances, source_variances, target_variances], dim=1)


                    
                    # Extract the variances (diagonal) as node features
                    # node_features = np.diag(cov_matrix)
                    # x = torch.tensor(node_features, dtype=torch.float)
                    # now every row of the cov matrix becomes part of the node features (the variance and all the associated covs of that company)
                    x = torch.tensor(cov_matrix, dtype=torch.float)

    
                    # Process the next covariance matrix to create target values
                    adj_matrix_next = next_covol_matrix.copy()
                    np.fill_diagonal(adj_matrix_next, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the next adjacency matrix (excluding diagonal)
                    # mask = np.triu(np.ones_like(adj_matrix_next), k=1) > 0
                    # y_edge = torch.tensor(adj_matrix_next[mask], dtype=torch.float)
    
                    # Extract the variances (diagonal) of the next covariance matrix as target node features
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float)
                    
                    
                    # Create PyTorch Geometric Data object
                    data = Data(x=x, 
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                # y_edge=y_edge, 
                                y_x=y_x)
                    
                    seq_data_list.append(data)


                    
                
                # Combine the sequence of Data objects into a single object with the desired format
                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=2).reshape(seq_data_list[0].x.shape[0],-1),
                                edge_index=seq_data_list[-1].edge_index,
                                edge_attr=torch.stack([d.edge_attr for d in seq_data_list], dim=2).reshape(seq_data_list[-1].edge_index.shape[1],-1),
                                # y_edge=seq_data_list[-1].y_edge,
                                y_x=seq_data_list[-1].y_x)
                data_list.append(seq_data)



        # Use ALL data - the train/test split will be handled during training
        # This ensures both GNN and LSTM have access to the same samples
        data, slices = self.collate(data_list)  # Use ALL data
        torch.save((data, slices), os.path.join(os.getcwd(),self.processed_paths[0]))



class CovarianceLaggedMultiOutputDataset(InMemoryDataset):
    def __init__(self, hdf5_file1, hdf5_file2, root='processed_data/cached_datasets_lagged_moutput/', transform=None, pre_transform=None, seq_length=None, future_steps=14):
        self.hdf5_file1 = hdf5_file1
        self.hdf5_file2 = hdf5_file2
        self.root = root
        self.seq_length = seq_length
        self.future_steps = future_steps
        super(CovarianceLaggedMultiOutputDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file1), os.path.basename(self.hdf5_file2)]


    @property
    def processed_file_names(self):
        return ['data_temp.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        # Open both HDF5 files simultaneously
        with h5py.File(self.hdf5_file1, 'r') as f1, h5py.File(self.hdf5_file2, 'r') as f2:
            keys = list(f1.keys())
            # order keys
            keys = natsorted(keys)
            ordered = all(int(keys[i]) <= int(keys[i+1]) for i in range(len(keys)-1))
            assert ordered, 'Keys of files 1 have not been ordered'
            keys = list(f2.keys())
            # order keys
            keys = natsorted(keys)
            ordered = all(int(keys[i]) <= int(keys[i+1]) for i in range(len(keys)-1))
            assert ordered, 'Keys of files 2 have not been ordered'
    
            for i in range(len(f1) - self.seq_length - self.future_steps + 1):
                seq_data_list = []
                for j in range(self.seq_length):
                    # vol data
                    cov_matrix = np.array(f1[keys[i+j]])
                    # volvol data
                    covol_matrix = np.array(f2[keys[i+j]])
                    
                    # Create the adjacency matrix from the covariance matrix
                    adj_matrix = covol_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Set the diagonal to zero
    
                    # # Create edge_index tensor
                    mask = np.triu(np.ones_like(adj_matrix), k=1) > 0
                    # edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
    
                    # # Extract the variances (diagonal) as a separate tensor
                    # variances = torch.tensor(np.diag(covol_matrix), dtype=torch.float)
                    # source_indices = edge_index[0]
                    # target_indices = edge_index[1]
                    # source_variances = variances[source_indices]
                    # target_variances = variances[target_indices]
                    # covariances = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    # edge_attr = torch.stack([covariances, source_variances, target_variances], dim=1)
                    # Create edge_index tensor for upper triangle
                    edge_index_upper = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
                    # Create edge_index tensor for lower triangle (transpose of upper)
                    edge_index_lower = edge_index_upper[[1, 0], :]
                    # Combine both to get full edge_index
                    edge_index = torch.cat([edge_index_upper, edge_index_lower], dim=1)

                    # edge_attr = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    # Extract the variances (diagonal) as a separate tensor
                    variances = torch.tensor(np.diag(covol_matrix), dtype=torch.float)
                    # Get the source and target indices from the edge_index
                    source_indices = edge_index[0]
                    target_indices = edge_index[1]
                    # Extract the variances for the source and target nodes
                    source_variances = variances[source_indices]
                    target_variances = variances[target_indices]
                    # Create the original edge attributes (covariances)
                    covariances = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    # Duplicate the covariances for the lower triangle
                    covariances = torch.cat([covariances, covariances])
                    # Concatenate the covariances with the source and target variances
                    edge_attr = torch.stack([covariances, source_variances, target_variances], dim=1)
                    
    
                    x = torch.tensor(cov_matrix, dtype=torch.float)
    
                    # Create a tensor to store the next 'self.future_steps' for each node
                    y_x_future_steps = []
                    for k in range(self.future_steps):
                        next_cov_matrix_k_steps = np.array(f1[keys[i+j+k+1]])
                        y_x_k_steps = torch.tensor(np.diag(next_cov_matrix_k_steps), dtype=torch.float)
                        y_x_future_steps.append(y_x_k_steps)
    
                    y_x_future_steps = torch.stack(y_x_future_steps, dim=1)  # Shape will be [num_nodes, self.future_steps]
    
                    # Create PyTorch Geometric Data object
                    data = Data(x=x, 
                                edge_index=edge_index,
                                edge_attr=edge_attr,
                                y_x=y_x_future_steps)
    
                    seq_data_list.append(data)
                    
    
                # Combine the sequence of Data objects into a single object with the desired format
                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=2).reshape(seq_data_list[0].x.shape[0],-1),
                                edge_index=seq_data_list[-1].edge_index,
                                edge_attr=torch.stack([d.edge_attr for d in seq_data_list], dim=2).reshape(seq_data_list[-1].edge_index.shape[1],-1),
                                y_x=seq_data_list[-1].y_x.reshape(-1))  # This will now be a [num_nodes, self.future_steps] tensor
    
                data_list.append(seq_data)


    
        # Use proper temporal split instead of hardcoded cutoff
        train_start_idx = max(0, len(data_list) - int(len(data_list) * 0.8))
        data, slices = self.collate(data_list[train_start_idx:])
        torch.save((data, slices), os.path.join(os.getcwd(),self.processed_paths[0]))


        
class CovarianceSparseDataset(InMemoryDataset):
    def __init__(self, hdf5_file, root='processed_data/cached_datasets_lagged/', transform=None, pre_transform=None, seq_length=None, threshold=None):
        self.hdf5_file = hdf5_file
        self.root = root
        self.seq_length = seq_length
        self.threshold = threshold
        super(CovarianceSparseDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file)]

    @property
    def processed_file_names(self):
        return ['data_temp.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        with h5py.File(self.hdf5_file, 'r') as f:
            keys = list(f.keys())
            # order keys
            keys = natsorted(keys)
            ordered = all(int(keys[i]) <= int(keys[i+1]) for i in range(len(keys)-1))
            assert ordered, 'Keys have not been ordered'
            # TODO cutoff hardcoded
            for i in range(len(f)-int(len(f)*0.2), len(f) - self.seq_length):
            # for i in range(len(f) - self.seq_length):
                seq_data_list = []
                for j in range(self.seq_length):
                    cov_matrix = np.array(f[keys[i+j]])
                    next_cov_matrix  = np.array(f[keys[i+j+1]])
                    assert int(keys[i+j]) + 1 == int(keys[i+j+1]), 'The labeling process is not considering consecutive matrices'
                    
                    # Create the adjacency matrix from the covariance matrix
                    adj_matrix = cov_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the adjacency matrix (excluding diagonal)
                    upper_triangle = np.triu(np.ones_like(adj_matrix), k=1)
                    # mask = upper_triangle.astype(bool) & (adj_matrix > 0)
                    mask = upper_triangle.astype(bool) & ((adj_matrix > self.threshold) | (adj_matrix < -self.threshold))

                    # Create edge_index tensor
                    edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()  
                      
                    edge_attr = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    

                    # Extract the variances (diagonal) as node features
                    node_features = np.diag(cov_matrix)
                    x = torch.tensor(node_features, dtype=torch.float)
    
                    # Process the next covariance matrix to create target values
                    adj_matrix_next = next_cov_matrix.copy()
                    np.fill_diagonal(adj_matrix_next, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the next adjacency matrix (excluding diagonal)
                    # mask = np.triu(np.ones_like(adj_matrix_next), k=1) > 0
                    # y_edge = torch.tensor(adj_matrix_next[mask], dtype=torch.float)
    
                    # Extract the variances (diagonal) of the next covariance matrix as target node features
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float)
                    
                    
                    # Create PyTorch Geometric Data object
                    data = Data(x=x, edge_index=edge_index, 
                                edge_attr=edge_attr,
                                # y_edge=y_edge, 
                                y_x=y_x)
                    
                    seq_data_list.append(data)
                    
                    
                
                # Combine the sequence of Data objects into a single object with the desired format
                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=1),
                                edge_index=seq_data_list[-1].edge_index,
                                edge_attr=seq_data_list[-1].edge_attr,
                                # y_edge=seq_data_list[-1].y_edge,
                                y_x=seq_data_list[-1].y_x)
                
                data_list.append(seq_data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
def check_reverse_edges_exist(edge_index):
    edge_index_t = edge_index.t().contiguous()
    
    # Convert to set for faster lookup
    edge_set = set(map(tuple, edge_index_t.tolist()))

    for edge in edge_set:
        if edge[::-1] not in edge_set:
            return False
            
    return True
        
        
        
# if not self.fully_connected:
#     upper_triangle = np.triu(np.ones_like(adj_matrix), k=1)
#     sparse_mask = upper_triangle.astype(bool) & (adj_matrix > 0)
#     sparse_edge_index = torch.tensor(np.argwhere(sparse_mask), dtype=torch.long).t().contiguous() 
    
#     fixed_value = -999
#     modified_edge_index = edge_index.clone()
#     for i in range(edge_index.size(1)):
#         edge = edge_index[:, i]
#         found = torch.any(torch.all(torch.eq(sparse_edge_index, edge.view(2, 1)), dim=0))
#         if not found:
#             modified_edge_index[:, i] = fixed_value
#     edge_index = modified_edge_index.clone()
#     # go back
#     # modified_edge_index[:, modified_edge_index[0] != -999]
    
#     sparse_edge_attr = torch.tensor(adj_matrix[sparse_mask], dtype=torch.float)
#     modified_edge_attr = edge_attr.clone()

#     for i in range(edge_attr.size(0)):
#         value = edge_attr[i]
#         if value.item() not in sparse_edge_attr.tolist():
#             modified_edge_attr[i] = fixed_value
#     # go back
#     # modified_edge_attr[modified_edge_attr!=-999]