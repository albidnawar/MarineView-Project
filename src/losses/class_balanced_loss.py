"""
Class-Balanced Loss implementation for handling extreme class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples.
    
    Addresses class imbalance by re-weighting the loss of each example based on 
    the effective number of samples for its class.
    
    CB_loss(p, y) = -1/E_n * log(p_y)
    where E_n = (1 - beta^n) / (1 - beta)
    
    Args:
        samples_per_class (list): Number of samples for each class
        beta (float): Hyperparameter for class balanced loss, typically 0.9999
        loss_type (str): Type of loss to use: 'focal' | 'sigmoid' | 'softmax'
        gamma (float): Focusing parameter for focal loss variant
    """
    
    def __init__(self, samples_per_class, beta=0.9999, loss_type='softmax', gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions from model, shape: (batch_size, num_classes)
            targets: Ground truth labels, shape: (batch_size,)
            
        Returns:
            loss: Computed class-balanced loss
        """
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
            
        if self.loss_type == 'focal':
            cb_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
            p_t = torch.exp(-cb_loss)
            cb_loss = (1 - p_t) ** self.gamma * cb_loss
            return cb_loss.mean()
        elif self.loss_type == 'sigmoid':
            # Binary cross-entropy variant
            pred = torch.sigmoid(inputs)
            cb_loss = F.binary_cross_entropy(pred, F.one_hot(targets, inputs.size(1)).float(), 
                                            weight=self.weights.unsqueeze(0), reduction='mean')
            return cb_loss
        else:  # softmax
            cb_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='mean')
            return cb_loss
