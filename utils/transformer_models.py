#!/usr/bin/env python3
"""
Transformer-based Models for Volatility Spillover Forecasting
==============================================================
Properly implemented TransformerConv models following PyTorch Geometric latest API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv, 
    global_mean_pool, 
    global_max_pool,
    global_add_pool,
    LayerNorm,
    BatchNorm
)
from torch_geometric.utils import softmax
import numpy as np
from typing import Optional, List, Tuple, Union
import math


class ImprovedTransformerConv(TransformerConv):
    """
    Enhanced TransformerConv with proper attention mechanism
    Following the exact PyG documentation specification
    """
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs
    ):
        """
        Initialize TransformerConv following PyG documentation
        
        Args:
            in_channels: Size of input features
            out_channels: Size of output features per head
            heads: Number of attention heads
            concat: Whether to concatenate or average heads
            beta: Whether to use learnable skip connection weight
            dropout: Dropout probability for attention coefficients
            edge_dim: Edge feature dimensionality
            bias: Whether to use bias
            root_weight: Whether to add transformed root features
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            beta=beta,
            dropout=dropout,
            edge_dim=edge_dim,
            bias=bias,
            root_weight=root_weight,
            **kwargs
        )
    
    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with proper attention weight handling
        """
        # Call parent forward method
        if return_attention_weights:
            out, (edge_index_att, alpha) = super().forward(
                x, edge_index, edge_attr, return_attention_weights=True
            )
            return out, (edge_index_att, alpha)
        else:
            return super().forward(x, edge_index, edge_attr)


class TransformerGNN(nn.Module):
    """
    Complete Transformer-based GNN for volatility forecasting
    Properly implements multi-head attention with edge features
    """
    
    def __init__(
        self,
        num_node_features: int = 31,
        num_edge_features: int = 1,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        edge_dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        activation: str = 'gelu',
        output_dim: int = 1,
        concat_heads: bool = True,
        beta: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.use_residual = use_residual
        self.concat_heads = concat_heads
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # Edge feature projection (project to match head dimension)
        if num_edge_features > 0:
            self.edge_proj = nn.Linear(num_edge_features, hidden_dim // num_heads)
        else:
            self.edge_proj = None
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Determine layer configuration
            is_last = (i == num_layers - 1)
            layer_heads = 1 if is_last else num_heads
            layer_concat = False if is_last else concat_heads
            
            # Input/output dimensions
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim if concat_heads else hidden_dim
            
            out_dim_per_head = hidden_dim // layer_heads if layer_concat else hidden_dim
            
            # Create TransformerConv layer
            self.transformer_layers.append(
                ImprovedTransformerConv(
                    in_channels=in_dim,
                    out_channels=out_dim_per_head,
                    heads=layer_heads,
                    concat=layer_concat,
                    beta=beta and not is_last,  # No beta for last layer
                    dropout=dropout,
                    edge_dim=hidden_dim // num_heads if self.edge_proj else None,
                    bias=True,
                    root_weight=True
                )
            )
            
            # Layer normalization
            norm_dim = hidden_dim if layer_concat or is_last else hidden_dim
            if use_layer_norm:
                self.norm_layers.append(LayerNorm(norm_dim))
            else:
                self.norm_layers.append(nn.Identity())
            
            # Feed-forward network (except for last layer)
            if not is_last:
                self.ff_layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 4),
                        self.activation,
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 4, hidden_dim),
                        nn.Dropout(dropout)
                    )
                )
        
        # Output projection
        final_dim = hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            return nn.GELU()
    
    def forward(
        self, 
        data,
        return_attention_weights: bool = False
    ):
        """
        Forward pass through the transformer GNN
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr
            return_attention_weights: Whether to return attention weights
        
        Returns:
            predictions: Node-level predictions
            attention_weights: (optional) List of attention weights per layer
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Input projection
        x = self.input_proj(x)
        
        # Project edge features if available
        if edge_attr is not None and self.edge_proj is not None:
            edge_attr_proj = self.edge_proj(edge_attr)
            
            # Apply edge dropout
            if self.training and self.edge_dropout > 0:
                edge_mask = torch.rand(edge_attr.size(0), device=edge_attr.device) > self.edge_dropout
                edge_attr_proj = edge_attr_proj * edge_mask.unsqueeze(-1)
        else:
            edge_attr_proj = None
        
        # Store attention weights if requested
        attention_weights = [] if return_attention_weights else None
        
        # Pass through transformer layers
        for i in range(self.num_layers):
            x_residual = x
            
            # Transformer convolution
            if return_attention_weights:
                x, (_, alpha) = self.transformer_layers[i](
                    x, edge_index, edge_attr_proj, 
                    return_attention_weights=True
                )
                attention_weights.append(alpha)
            else:
                x = self.transformer_layers[i](x, edge_index, edge_attr_proj)
            
            # Normalization and activation
            x = self.norm_layers[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Feed-forward network (if not last layer)
            if i < self.num_layers - 1:
                x_ff = self.ff_layers[i](x)
                
                # Residual connection
                if self.use_residual and x_ff.shape == x_residual.shape:
                    x = x_ff + x_residual
                else:
                    x = x_ff
        
        # Output projection
        out = self.output_proj(x)
        
        if return_attention_weights:
            return out, attention_weights
        return out


class TransformerSpillNet(nn.Module):
    """
    Specialized Transformer for modeling volatility spillovers
    Combines TransformerConv with spillover-specific mechanisms
    """
    
    def __init__(
        self,
        num_node_features: int = 31,
        num_edge_features: int = 1,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        edge_dropout: float = 0.05,
        activation: str = 'gelu',
        output_dim: int = 1,
        use_temporal_encoding: bool = True,
        aggregation: str = 'multi'  # 'mean', 'max', 'multi'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        self.use_temporal_encoding = use_temporal_encoding
        self.aggregation = aggregation
        
        # Activation
        self.activation = nn.GELU() if activation == 'gelu' else nn.ELU()
        
        # Input encoder with enhanced features
        self.input_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            LayerNorm(hidden_dim),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # Edge encoder for covolatility
        if num_edge_features > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(num_edge_features, hidden_dim // num_heads),
                LayerNorm(hidden_dim // num_heads),
                self.activation
            )
        else:
            self.edge_encoder = None
        
        # Main transformer layers
        self.transformer_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Alternate between different head configurations
            if i == 0:
                in_dim = hidden_dim
                layer_heads = num_heads
                concat = True
            elif i == num_layers - 1:
                in_dim = hidden_dim
                layer_heads = 1
                concat = False
            else:
                in_dim = hidden_dim
                layer_heads = num_heads
                concat = True
            
            # Create transformer layer
            self.transformer_layers.append(
                ImprovedTransformerConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim // layer_heads if concat else hidden_dim,
                    heads=layer_heads,
                    concat=concat,
                    beta=True,
                    dropout=dropout,
                    edge_dim=hidden_dim // num_heads if self.edge_encoder else None,
                    bias=True,
                    root_weight=True
                )
            )
            
            # Layer normalization
            self.norm_layers.append(LayerNorm(hidden_dim))
        
        # Temporal encoding module
        if use_temporal_encoding:
            self.temporal_encoder = TemporalSpilloverEncoder(
                d_model=hidden_dim,
                n_heads=num_heads,
                d_ff=hidden_dim * 2,
                dropout=dropout
            )
        
        # Determine pooling dimension
        if aggregation == 'multi':
            pool_dim = hidden_dim * 3
        else:
            pool_dim = hidden_dim
        
        # Spillover intensity predictor
        self.spillover_predictor = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            LayerNorm(hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.activation,
            nn.Dropout(dropout)
        )
        
        # Output network
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            LayerNorm(hidden_dim // 4),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, data, return_spillover_weights: bool = False):
        """
        Forward pass through TransformerSpillNet
        
        Args:
            data: PyG Data object
            return_spillover_weights: Whether to return attention weights
        
        Returns:
            predictions: Volatility predictions
            spillover_weights: (optional) Attention weights showing spillover patterns
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )
        
        # Encode inputs
        x = self.input_encoder(x)
        
        # Encode edge features
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr_encoded = self.edge_encoder(edge_attr)
            
            # Edge dropout
            if self.training and self.edge_dropout > 0:
                edge_mask = torch.rand(edge_attr.size(0), device=edge_attr.device) > self.edge_dropout
                edge_attr_encoded = edge_attr_encoded * edge_mask.unsqueeze(-1)
        else:
            edge_attr_encoded = None
        
        # Store attention weights
        spillover_weights = [] if return_spillover_weights else None
        
        # Process through transformer layers
        for i in range(self.num_layers):
            x_residual = x
            
            # Transformer convolution
            if return_spillover_weights:
                x, (_, alpha) = self.transformer_layers[i](
                    x, edge_index, edge_attr_encoded,
                    return_attention_weights=True
                )
                spillover_weights.append(alpha)
            else:
                x = self.transformer_layers[i](x, edge_index, edge_attr_encoded)
            
            # Normalization and activation
            x = self.norm_layers[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if x.shape == x_residual.shape:
                x = x + x_residual * 0.5
        
        # Temporal encoding if available
        if self.use_temporal_encoding and hasattr(data, 'temporal_x'):
            x = self.temporal_encoder(x, batch)
        
        # Multi-scale pooling
        if self.aggregation == 'multi':
            x_pooled = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch)
            ], dim=-1)
        elif self.aggregation == 'mean':
            x_pooled = global_mean_pool(x, batch)
        else:  # max
            x_pooled = global_max_pool(x, batch)
        
        # Predict spillover intensity
        spillover_intensity = self.spillover_predictor(x_pooled)
        
        # Generate final predictions
        out = self.output_network(spillover_intensity)
        
        # Expand to all nodes if needed
        if out.size(0) != x.size(0):
            out = out[batch]
        
        if return_spillover_weights and spillover_weights:
            # Average attention weights across layers
            avg_weights = torch.stack(spillover_weights).mean(dim=0)
            return out, avg_weights
        
        return out


class TemporalSpilloverEncoder(nn.Module):
    """
    Temporal encoding module for spillover patterns
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        # Multi-head attention for temporal patterns
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, batch):
        """Apply temporal encoding"""
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal information
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding"""
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        if x.dim() == 2:
            return x + self.pe[:seq_len]
        else:
            return x + self.pe[:seq_len].unsqueeze(0)


def create_transformer_model(
    model_type: str = 'transformer',
    **kwargs
) -> nn.Module:
    """
    Factory function to create transformer models
    
    Args:
        model_type: 'transformer' or 'spillnet'
        **kwargs: Model configuration
    
    Returns:
        Initialized model
    """
    if model_type == 'transformer':
        return TransformerGNN(**kwargs)
    elif model_type == 'spillnet':
        return TransformerSpillNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    from torch_geometric.data import Data, Batch
    
    print("="*80)
    print("Testing Transformer Models for Volatility Forecasting")
    print("="*80)
    
    # Create synthetic data
    num_nodes = 30  # 30 stocks
    num_features = 31
    batch_size = 4
    
    # Create batch of graphs
    graphs = []
    for b in range(batch_size):
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        edge_attr = torch.randn(edge_index.size(1), 1)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(data)
    
    batch = Batch.from_data_list(graphs)
    
    # Test TransformerGNN
    print("\n1. Testing TransformerGNN...")
    model1 = TransformerGNN(
        num_node_features=31,
        num_edge_features=1,
        hidden_dim=256,
        num_heads=8,
        num_layers=3
    )
    
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    with torch.no_grad():
        output = model1(batch)
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Test with attention weights
        output, attention = model1(batch, return_attention_weights=True)
        print(f"   Number of attention layers: {len(attention)}")
    
    # Test TransformerSpillNet
    print("\n2. Testing TransformerSpillNet...")
    model2 = TransformerSpillNet(
        num_node_features=31,
        num_edge_features=1,
        hidden_dim=256,
        num_heads=8,
        num_layers=4
    )
    
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    with torch.no_grad():
        output = model2(batch)
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Test with spillover weights
        output, spillover = model2(batch, return_spillover_weights=True)
        print(f"   Spillover weights shape: {spillover.shape}")
    
    print("\n" + "="*80)
    print("✅ All Transformer models tested successfully!")
    print("="*80)