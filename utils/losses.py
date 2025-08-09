#!/usr/bin/env python3
"""
Loss Functions for Volatility Forecasting
==========================================
Custom loss functions optimized for financial volatility prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class QLIKELoss(nn.Module):
    """
    Quasi-Likelihood Loss (QLIKE) for variance forecasting
    
    QLIKE is an asymmetric loss function that penalizes under-prediction
    more heavily than over-prediction, which is economically sensible for
    risk management (better to overestimate risk than underestimate).
    
    Formula: L(σ², σ̂²) = σ̂²/σ² - ln(σ̂²/σ²) - 1
    
    Properties:
    - Asymmetric: penalizes under-prediction more
    - Scale-free: consistent across different volatility levels
    - Robust: works well with heavy-tailed distributions
    """
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate QLIKE loss
        
        Args:
            predictions: Predicted variance values (σ²)
            targets: True variance values (σ²)
        
        Returns:
            Scalar loss value
        """
        # Add epsilon for numerical stability
        predictions = predictions + self.epsilon
        targets = targets + self.epsilon
        
        # Calculate ratio
        ratio = predictions / targets
        
        # QLIKE formula
        qlike = ratio - torch.log(ratio) - 1
        
        # Return mean loss
        return torch.mean(qlike)


class RobustQLIKELoss(nn.Module):
    """
    Robust version of QLIKE with outlier handling
    """
    
    def __init__(self, epsilon: float = 1e-8, clip_ratio: float = 10.0):
        super().__init__()
        self.epsilon = epsilon
        self.clip_ratio = clip_ratio
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate robust QLIKE loss with ratio clipping
        """
        # Ensure positive values by taking absolute value and adding epsilon
        # This is valid since we're dealing with variance which is always positive
        predictions = torch.abs(predictions) + self.epsilon
        targets = torch.abs(targets) + self.epsilon
        
        # Calculate ratio with clipping
        ratio = predictions / targets
        ratio = torch.clamp(ratio, 1/self.clip_ratio, self.clip_ratio)
        
        # QLIKE formula with NaN protection
        log_ratio = torch.log(ratio)
        
        # Check for NaN/Inf
        if torch.isnan(log_ratio).any() or torch.isinf(log_ratio).any():
            # Fallback to a safe value
            log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=np.log(self.clip_ratio), neginf=-np.log(self.clip_ratio))
        
        qlike = ratio - log_ratio - 1
        
        # Final NaN check
        if torch.isnan(qlike).any():
            # Return a large but finite loss value
            return torch.tensor(10.0, device=predictions.device, requires_grad=True)
        
        return torch.mean(qlike)


class MSEVolatilityLoss(nn.Module):
    """
    Mean Squared Error loss for volatility (not variance)
    Includes option to weight by realized volatility
    """
    
    def __init__(self, weighted: bool = False):
        super().__init__()
        self.weighted = weighted
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate MSE loss, optionally weighted by volatility level
        """
        # Standard MSE
        mse = (predictions - targets) ** 2
        
        if self.weighted:
            # Weight by volatility level (higher vol = more weight)
            weights = targets / (torch.mean(targets) + 1e-8)
            mse = mse * weights
        
        return torch.mean(mse)


class HuberVolatilityLoss(nn.Module):
    """
    Huber loss (smooth L1) for robust volatility forecasting
    Less sensitive to outliers than MSE
    """
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Huber loss
        """
        return F.smooth_l1_loss(predictions, targets, beta=self.delta)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss that penalizes under/over prediction differently
    Useful for risk-averse forecasting
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: Weight for under-prediction (1-alpha for over-prediction)
                   alpha > 0.5 penalizes under-prediction more
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate asymmetric loss
        """
        errors = predictions - targets
        
        # Different weights for positive/negative errors
        weights = torch.where(errors < 0, self.alpha, 1 - self.alpha)
        
        # Weighted squared errors
        loss = weights * (errors ** 2)
        
        return torch.mean(loss)


class CombinedVolatilityLoss(nn.Module):
    """
    Combines multiple loss functions for robust training
    """
    
    def __init__(self, 
                 qlike_weight: float = 0.7,
                 mse_weight: float = 0.2,
                 huber_weight: float = 0.1,
                 epsilon: float = 1e-8):
        super().__init__()
        self.qlike_weight = qlike_weight
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        
        # Initialize component losses
        self.qlike = QLIKELoss(epsilon=epsilon)
        self.mse = MSEVolatilityLoss()
        self.huber = HuberVolatilityLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss
        """
        loss = 0.0
        
        if self.qlike_weight > 0:
            loss += self.qlike_weight * self.qlike(predictions, targets)
        
        if self.mse_weight > 0:
            loss += self.mse_weight * self.mse(predictions, targets)
        
        if self.huber_weight > 0:
            loss += self.huber_weight * self.huber(predictions, targets)
        
        return loss


class SpilloverLoss(nn.Module):
    """
    Specialized loss for SpillNet that includes spillover regularization
    """
    
    def __init__(self,
                 base_loss: str = 'qlike',
                 spillover_reg: float = 0.01,
                 temporal_smooth: float = 0.01,
                 epsilon: float = 1e-8):
        super().__init__()
        
        # Base loss function
        if base_loss == 'qlike':
            self.base_loss = QLIKELoss(epsilon=epsilon)
        elif base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        elif base_loss == 'huber':
            self.base_loss = HuberVolatilityLoss()
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
        
        self.spillover_reg = spillover_reg
        self.temporal_smooth = temporal_smooth
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor,
                spillover_matrix: Optional[torch.Tensor] = None,
                temporal_diff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss with spillover regularization
        
        Args:
            predictions: Predicted volatilities
            targets: True volatilities
            spillover_matrix: Optional spillover attention weights for regularization
            temporal_diff: Optional temporal differences for smoothness
        
        Returns:
            Total loss
        """
        # Base prediction loss
        loss = self.base_loss(predictions, targets)
        
        # Spillover regularization (encourage sparsity)
        if spillover_matrix is not None and self.spillover_reg > 0:
            # L1 regularization on spillover weights
            spillover_penalty = torch.mean(torch.abs(spillover_matrix))
            loss = loss + self.spillover_reg * spillover_penalty
        
        # Temporal smoothness regularization
        if temporal_diff is not None and self.temporal_smooth > 0:
            # Penalize large temporal jumps
            temporal_penalty = torch.mean(temporal_diff ** 2)
            loss = loss + self.temporal_smooth * temporal_penalty
        
        return loss


def combined_loss(predictions: torch.Tensor, 
                  targets: torch.Tensor,
                  qlike_weight: float = 0.7,
                  mse_weight: float = 0.3,
                  epsilon: float = 1e-8) -> torch.Tensor:
    """
    Simple combined loss function (functional interface)
    
    Args:
        predictions: Predicted values
        targets: True values
        qlike_weight: Weight for QLIKE component
        mse_weight: Weight for MSE component
        epsilon: Small value for numerical stability
    
    Returns:
        Combined loss value
    """
    # QLIKE component
    qlike_loss = QLIKELoss(epsilon=epsilon)
    qlike_val = qlike_loss(predictions, targets)
    
    # MSE component
    mse_val = F.mse_loss(predictions, targets)
    
    # Combine
    return qlike_weight * qlike_val + mse_weight * mse_val


def get_loss_function(loss_type: str = 'qlike', **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name
    
    Args:
        loss_type: Type of loss ('qlike', 'mse', 'huber', 'combined', 'spillover')
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function module
    """
    if loss_type == 'qlike':
        return QLIKELoss(**kwargs)
    elif loss_type == 'robust_qlike':
        return RobustQLIKELoss(**kwargs)
    elif loss_type == 'mse':
        return MSEVolatilityLoss(**kwargs)
    elif loss_type == 'huber':
        return HuberVolatilityLoss(**kwargs)
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedVolatilityLoss(**kwargs)
    elif loss_type == 'spillover':
        return SpilloverLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    """Test loss functions"""
    print("Testing volatility loss functions...")
    print("="*60)
    
    # Create synthetic data
    batch_size = 32
    num_assets = 30
    
    # Predictions and targets (variance scale)
    predictions = torch.rand(batch_size, num_assets) * 0.01  # Variance scale
    targets = torch.rand(batch_size, num_assets) * 0.01
    
    # Test each loss function
    losses = {
        'QLIKE': QLIKELoss(),
        'Robust QLIKE': RobustQLIKELoss(),
        'MSE': MSEVolatilityLoss(),
        'Weighted MSE': MSEVolatilityLoss(weighted=True),
        'Huber': HuberVolatilityLoss(),
        'Asymmetric': AsymmetricLoss(alpha=0.7),
        'Combined': CombinedVolatilityLoss(),
        'Spillover': SpilloverLoss()
    }
    
    for name, loss_fn in losses.items():
        if name == 'Spillover':
            # Test with spillover matrix
            spillover_matrix = torch.rand(batch_size, num_assets, num_assets)
            loss_val = loss_fn(predictions, targets, spillover_matrix=spillover_matrix)
        else:
            loss_val = loss_fn(predictions, targets)
        
        print(f"{name:15s}: {loss_val.item():.6f}")
    
    # Test combined loss function
    combined_val = combined_loss(predictions, targets)
    print(f"{'Combined (fn)':15s}: {combined_val.item():.6f}")
    
    print("\n" + "="*60)
    print("✅ All loss functions tested successfully!")