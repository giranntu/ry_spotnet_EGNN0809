#!/usr/bin/env python3
"""
Advanced GNN Models for Volatility Forecasting
==============================================
Drop-in replacements and improvements for SpotV2Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, TransformerConv, PNAConv, GINEConv, 
    SuperGATConv, GPSConv, GENConv,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm, LayerNorm, GraphNorm
)
from torch_geometric.utils import degree
from typing import Optional, List, Tuple


class GATv2Model(nn.Module):
    """
    Improved GAT using GATv2Conv (fixes static attention problem)
    Direct upgrade from original GATModel
    """
    def __init__(self, 
                 num_node_features: int,
                 num_edge_features: int,
                 num_heads: int = 8,
                 output_node_channels: int = 1,
                 dim_hidden_layers: List[int] = [256, 128],
                 dropout_att: float = 0.1,
                 dropout: float = 0.2,
                 activation: str = 'elu',
                 negative_slope: float = 0.2,
                 normalize: bool = True):
        super().__init__()
        
        self.dropout = dropout
        self.activation_name = activation
        
        # Choose activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope)
        
        # GATv2 layers with edge features
        self.conv1 = GATv2Conv(
            num_node_features, 
            dim_hidden_layers[0] // num_heads,
            heads=num_heads,
            dropout=dropout_att,
            edge_dim=num_edge_features,
            concat=True,
            negative_slope=negative_slope,
            share_weights=True  # GATv2 specific
        )
        
        self.norm1 = BatchNorm(dim_hidden_layers[0])
        
        self.conv2 = GATv2Conv(
            dim_hidden_layers[0],
            dim_hidden_layers[1] // num_heads,
            heads=num_heads,
            dropout=dropout_att,
            edge_dim=num_edge_features,
            concat=True,
            negative_slope=negative_slope,
            share_weights=True
        )
        
        self.norm2 = BatchNorm(dim_hidden_layers[1])
        
        # Output layer
        self.conv3 = GATv2Conv(
            dim_hidden_layers[1],
            output_node_channels,
            heads=1,
            dropout=dropout_att,
            edge_dim=num_edge_features,
            concat=False
        )
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output
        x = self.conv3(x, edge_index, edge_attr)
        
        return x


class TransformerGNNModel(nn.Module):
    """
    Transformer-based GNN for volatility forecasting
    Best for capturing complex financial correlations
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 num_heads: int = 8,
                 output_node_channels: int = 1,
                 dim_hidden_layers: List[int] = [256, 128],
                 dropout: float = 0.2,
                 beta: bool = True):  # Use beta for multi-head concat
        super().__init__()
        
        self.dropout = dropout
        
        # Transformer layers
        self.conv1 = TransformerConv(
            num_node_features,
            dim_hidden_layers[0] // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=num_edge_features,
            beta=beta,
            concat=True
        )
        
        self.norm1 = LayerNorm(dim_hidden_layers[0])
        
        self.conv2 = TransformerConv(
            dim_hidden_layers[0],
            dim_hidden_layers[1] // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=num_edge_features,
            beta=beta,
            concat=True
        )
        
        self.norm2 = LayerNorm(dim_hidden_layers[1])
        
        # Output with single head
        self.conv3 = TransformerConv(
            dim_hidden_layers[1],
            output_node_channels,
            heads=1,
            dropout=dropout,
            edge_dim=num_edge_features,
            beta=False,
            concat=False
        )
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output
        x = self.conv3(x, edge_index, edge_attr)
        
        return x


class PNAModel(nn.Module):
    """
    Principal Neighbourhood Aggregation GNN
    Designed for continuous features and regression tasks
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 output_node_channels: int = 1,
                 dim_hidden_layers: List[int] = [256, 128],
                 dropout: float = 0.2,
                 aggregators: List[str] = ['mean', 'max', 'min', 'std'],
                 scalers: List[str] = ['identity', 'amplification', 'attenuation'],
                 deg: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.dropout = dropout
        
        # Compute degree if not provided (for 30 stocks, fully connected)
        if deg is None:
            deg = torch.ones(30) * 29  # Each node connects to 29 others
        
        # PNA layers
        self.conv1 = PNAConv(
            num_node_features,
            dim_hidden_layers[0],
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=num_edge_features,
            towers=1,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )
        
        self.norm1 = BatchNorm(dim_hidden_layers[0])
        
        self.conv2 = PNAConv(
            dim_hidden_layers[0],
            dim_hidden_layers[1],
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=num_edge_features,
            towers=1,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )
        
        self.norm2 = BatchNorm(dim_hidden_layers[1])
        
        # Output layer
        self.fc_out = nn.Linear(dim_hidden_layers[1], output_node_channels)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output
        x = self.fc_out(x)
        
        return x


class GINEModel(nn.Module):
    """
    Graph Isomorphism Network with Edge features
    Maximum expressive power for graph learning
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 output_node_channels: int = 1,
                 dim_hidden_layers: List[int] = [256, 128],
                 dropout: float = 0.2,
                 train_eps: bool = True):
        super().__init__()
        
        self.dropout = dropout
        
        # MLP for first GINE layer
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, dim_hidden_layers[0]),
            nn.BatchNorm1d(dim_hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(dim_hidden_layers[0], dim_hidden_layers[0])
        )
        
        self.conv1 = GINEConv(nn1, train_eps=train_eps, edge_dim=num_edge_features)
        self.norm1 = BatchNorm(dim_hidden_layers[0])
        
        # MLP for second GINE layer
        nn2 = nn.Sequential(
            nn.Linear(dim_hidden_layers[0], dim_hidden_layers[1]),
            nn.BatchNorm1d(dim_hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(dim_hidden_layers[1], dim_hidden_layers[1])
        )
        
        self.conv2 = GINEConv(nn2, train_eps=train_eps, edge_dim=num_edge_features)
        self.norm2 = BatchNorm(dim_hidden_layers[1])
        
        # Output
        self.fc_out = nn.Linear(dim_hidden_layers[1], output_node_channels)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output
        x = self.fc_out(x)
        
        return x


class HybridTemporalGNN(nn.Module):
    """
    Combines GNN with LSTM for temporal volatility forecasting
    Can use any GNN backbone
    """
    def __init__(self,
                 gnn_model: str = 'gatv2',  # 'gatv2', 'transformer', 'pna', 'gine'
                 num_node_features: int = 31,
                 num_edge_features: int = 1,
                 hidden_dim: int = 256,
                 lstm_hidden: int = 128,
                 num_lstm_layers: int = 2,
                 output_channels: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        
        # Choose GNN backbone
        if gnn_model == 'gatv2':
            self.gnn = GATv2Model(
                num_node_features, num_edge_features,
                output_node_channels=hidden_dim,
                dim_hidden_layers=[hidden_dim, hidden_dim],
                dropout=dropout
            )
        elif gnn_model == 'transformer':
            self.gnn = TransformerGNNModel(
                num_node_features, num_edge_features,
                output_node_channels=hidden_dim,
                dim_hidden_layers=[hidden_dim, hidden_dim],
                dropout=dropout
            )
        elif gnn_model == 'pna':
            self.gnn = PNAModel(
                num_node_features, num_edge_features,
                output_node_channels=hidden_dim,
                dim_hidden_layers=[hidden_dim, hidden_dim],
                dropout=dropout
            )
        elif gnn_model == 'gine':
            self.gnn = GINEModel(
                num_node_features, num_edge_features,
                output_node_channels=hidden_dim,
                dim_hidden_layers=[hidden_dim, hidden_dim],
                dropout=dropout
            )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim,
            lstm_hidden,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Linear(lstm_hidden, output_channels)
        
    def forward(self, data_sequence):
        """
        Process sequence of graphs
        data_sequence: List of Data objects for each timestep
        """
        # Process each graph through GNN
        gnn_outputs = []
        for data in data_sequence:
            gnn_out = self.gnn(data)  # [num_nodes, hidden_dim]
            # Pool to get graph-level representation
            graph_repr = global_mean_pool(gnn_out, data.batch)
            gnn_outputs.append(graph_repr)
        
        # Stack for LSTM
        x = torch.stack(gnn_outputs, dim=1)  # [batch, seq_len, hidden_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep
        out = self.fc_out(lstm_out[:, -1, :])
        
        return out


def create_advanced_model(model_type: str = 'gatv2', **kwargs):
    """
    Factory function to create advanced GNN models
    
    Args:
        model_type: One of 'gatv2', 'transformer', 'pna', 'gine', 'hybrid'
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized model
    """
    if model_type == 'gatv2':
        return GATv2Model(**kwargs)
    elif model_type == 'transformer':
        return TransformerGNNModel(**kwargs)
    elif model_type == 'pna':
        return PNAModel(**kwargs)
    elif model_type == 'gine':
        return GINEModel(**kwargs)
    elif model_type == 'hybrid':
        return HybridTemporalGNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    from torch_geometric.data import Data
    
    # Create dummy data
    x = torch.randn(30, 31)  # 30 stocks, 31 features
    edge_index = torch.combinations(torch.arange(30), r=2).t()  # Fully connected
    edge_attr = torch.randn(edge_index.size(1), 1)  # Edge features
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    print("Testing GNN models for volatility forecasting...")
    
    # Test each model
    models = {
        'GATv2': GATv2Model(31, 1),
        'Transformer': TransformerGNNModel(31, 1),
        'PNA': PNAModel(31, 1),
        'GINE': GINEModel(31, 1)
    }
    
    for name, model in models.items():
        print(f"\n{name} Model:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Forward pass
        with torch.no_grad():
            out = model(data)
            print(f"  Output shape: {out.shape}")
            print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    
    print("\n✅ All models tested successfully!")