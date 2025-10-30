"""
PyTorch Dataset for acoustic marine mammal data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class AcousticDataset(Dataset):
    """
    Dataset for marine mammal acoustic signals.
    
    Args:
        data_paths (list): List of paths to preprocessed spectrograms or audio files
        labels (list): List of corresponding labels (species IDs)
        preprocessor: AudioPreprocessor instance for on-the-fly processing
        augmentation: AcousticAugmentation instance for data augmentation
        mode (str): 'train', 'val', or 'test'
        return_pairs (bool): If True, return pairs of augmented views (for SimCLR)
    """
    
    def __init__(self, data_paths, labels=None, preprocessor=None, 
                 augmentation=None, mode='train', return_pairs=False):
        self.data_paths = data_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.mode = mode
        self.return_pairs = return_pairs
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Returns:
            If return_pairs=True: ((view1, view2), label) for contrastive learning
            Otherwise: (spectrogram, label) for supervised learning
        """
        data_path = self.data_paths[idx]
        
        # Load or preprocess data
        if self.preprocessor is not None:
            # Process from audio file
            if isinstance(data_path, str) and os.path.exists(data_path):
                spec = self.preprocessor.preprocess(data_path)
            else:
                # Assume it's already a tensor or array
                spec = data_path
                if not isinstance(spec, torch.Tensor):
                    spec = torch.from_numpy(spec).float()
        else:
            # Load preprocessed spectrogram
            if isinstance(data_path, str):
                spec = torch.load(data_path)
            else:
                spec = data_path
                if not isinstance(spec, torch.Tensor):
                    spec = torch.from_numpy(spec).float()
        
        # Add channel dimension if needed
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        
        # Apply augmentation and return pairs for contrastive learning
        if self.return_pairs:
            if self.augmentation is not None:
                view1, view2 = self.augmentation.create_pair(spec)
            else:
                view1, view2 = spec, spec.clone()
            
            if self.labels is not None:
                return (view1, view2), self.labels[idx]
            else:
                return (view1, view2), -1  # Dummy label for unsupervised
        
        # Apply augmentation for regular training
        if self.mode == 'train' and self.augmentation is not None:
            spec = self.augmentation(spec)
        
        if self.labels is not None:
            return spec, self.labels[idx]
        else:
            return spec, -1  # Dummy label


class ImbalancedSampler:
    """
    Sampler that handles class imbalance by oversampling minority classes.
    
    Args:
        labels (list or array): Class labels for each sample
        oversample_factor (float): Factor to oversample minority classes
    """
    
    def __init__(self, labels, oversample_factor=2.0):
        self.labels = np.array(labels)
        self.oversample_factor = oversample_factor
        
        # Calculate class weights
        unique_classes, class_counts = np.unique(self.labels, return_counts=True)
        self.class_weights = 1.0 / class_counts
        self.class_weights = self.class_weights / self.class_weights.sum()
        
        # Create sample weights
        self.sample_weights = np.array([
            self.class_weights[np.where(unique_classes == label)[0][0]] 
            for label in self.labels
        ])
        
        # Scale weights for oversampling
        self.sample_weights = self.sample_weights * oversample_factor
        
    def get_weights(self):
        """Return sample weights for WeightedRandomSampler"""
        return self.sample_weights
