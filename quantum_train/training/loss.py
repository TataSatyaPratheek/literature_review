"""Loss functions for Quantum-Train."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy loss for classification.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize cross-entropy loss.
        
        Args:
            reduction: Specifies reduction to apply ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Model predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)
            
        Returns:
            Loss value
        """
        return F.cross_entropy(predictions, targets, reduction=self.reduction)

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Useful for difficult classification tasks.
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor in range (0,1)
            gamma: Exponent of modulating factor (1 - p_t)
            reduction: Specifies reduction to apply
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Compute focal loss.
        
        Args:
            predictions: Model predictions (logits), shape (N, C)
            targets: Ground truth labels, shape (N,)
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(loss_name='cross_entropy', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    loss_functions = {
        'cross_entropy': CrossEntropyLoss,
        'focal': FocalLoss
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)
