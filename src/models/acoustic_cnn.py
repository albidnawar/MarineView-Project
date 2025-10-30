"""
Acoustic CNN architecture for marine mammal acoustic feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticCNN(nn.Module):
    """
    Convolutional Neural Network for acoustic feature extraction.
    
    This CNN is designed to process mel-spectrogram representations of 
    marine mammal acoustic signals and extract meaningful features for 
    species classification.
    
    Args:
        input_channels (int): Number of input channels (1 for mono audio)
        num_features (int): Dimension of output feature vector
    """
    
    def __init__(self, input_channels=1, num_features=512):
        super(AcousticCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(512, num_features)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
               Expected shape for mel-spectrogram: (batch, 1, n_mels, time_steps)
            
        Returns:
            features: Extracted feature vector (batch_size, num_features)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Dropout and FC
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization (e.g., Grad-CAM).
        
        Args:
            x: Input tensor
            
        Returns:
            features_dict: Dictionary containing feature maps at different layers
        """
        features = {}
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        features['conv1'] = x
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        features['conv2'] = x
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        features['conv3'] = x
        x = self.pool(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        features['conv4'] = x
        x = self.pool(x)
        
        return features
