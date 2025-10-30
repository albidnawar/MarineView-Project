"""
Rare Species Classifier combining pre-trained encoder with classification head.
"""

import torch
import torch.nn as nn


class RareSpeciesClassifier(nn.Module):
    """
    Classifier for rare marine mammal species detection.
    
    Uses a pre-trained encoder (from SimCLR self-supervised learning) 
    and adds a classification head for species identification.
    
    Args:
        encoder: Pre-trained encoder network (e.g., from SimCLR)
        num_classes (int): Number of species classes to classify
        feature_dim (int): Dimension of encoder output features
        freeze_encoder (bool): Whether to freeze encoder weights during training
    """
    
    def __init__(self, encoder, num_classes, feature_dim=512, freeze_encoder=False):
        super(RareSpeciesClassifier, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            logits: Class predictions (batch_size, num_classes)
        """
        # Extract features from encoder
        features = self.encoder(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x):
        """Extract features for analysis"""
        return self.encoder(x)
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def freeze_encoder(self):
        """Freeze encoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = False
