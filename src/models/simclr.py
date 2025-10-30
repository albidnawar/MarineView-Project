"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
Adapted for acoustic data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLR(nn.Module):
    """
    SimCLR model for self-supervised contrastive learning.
    
    This implementation adapts SimCLR for acoustic marine mammal data,
    reducing the need for labeled data by learning representations from 
    unlabeled acoustic signals.
    
    Args:
        base_encoder: Base CNN encoder network
        projection_dim (int): Dimension of projection head output
        hidden_dim (int): Hidden dimension in projection head
    """
    
    def __init__(self, base_encoder, projection_dim=128, hidden_dim=512):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        
        # Get the feature dimension from encoder
        self.feature_dim = self._get_encoder_dim()
        
        # Projection head: MLP with one hidden layer
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def _get_encoder_dim(self):
        """Get output dimension of encoder"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 128, 128)  # Example acoustic input
            output = self.encoder(dummy_input)
            return output.shape[1]
    
    def forward(self, x):
        """
        Forward pass through encoder and projection head.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            features: Encoded features from base encoder
            projections: Projected features for contrastive learning
        """
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections
    
    def get_features(self, x):
        """Extract features without projection (for downstream tasks)"""
        return self.encoder(x)


class NT_Xent(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    The contrastive loss function used in SimCLR.
    
    Args:
        temperature (float): Temperature parameter for scaling
        batch_size (int): Size of batch
    """
    
    def __init__(self, temperature=0.5, batch_size=32):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss for a batch of positive pairs.
        
        Args:
            z_i: Projections of augmented view i (batch_size, projection_dim)
            z_j: Projections of augmented view j (batch_size, projection_dim)
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = z_i.shape[0]
        
        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate all projections
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )
        
        # Create mask to exclude self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Create positive pairs mask
        positives = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z_i.device)
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute loss
        labels = positives.long()
        loss = self.criterion(similarity_matrix, labels)
        loss = loss / (2 * batch_size)
        
        return loss
