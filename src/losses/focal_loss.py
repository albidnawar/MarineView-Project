"""
Focal Loss implementation for handling class imbalance in rare species detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: Addressing class imbalance by down-weighting easy examples.
    
    Loss(x, class) = -alpha * (1 - softmax(x)[class])^gamma * log(softmax(x)[class])
    
    Args:
        alpha (float or list): Weighting factor in [0, 1] to balance positive/negative examples.
                               Can be a list of per-class weights.
        gamma (float): Focusing parameter for modulating loss. gamma >= 0.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions from model (before softmax), shape: (batch_size, num_classes)
            targets: Ground truth labels, shape: (batch_size,)
            
        Returns:
            loss: Computed focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
