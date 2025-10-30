"""
Data augmentation techniques for acoustic signals in self-supervised learning.
"""

import torch
import numpy as np
import librosa


class AcousticAugmentation:
    """
    Augmentation strategies for acoustic data in contrastive learning.
    
    Implements various augmentations suitable for marine mammal vocalizations
    to create positive pairs for SimCLR training.
    
    Args:
        time_mask_param (int): Maximum possible length of time mask
        freq_mask_param (int): Maximum possible length of frequency mask
        noise_factor (float): Standard deviation of Gaussian noise
    """
    
    def __init__(self, time_mask_param=20, freq_mask_param=20, noise_factor=0.005):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_factor = noise_factor
        
    def time_mask(self, spec):
        """
        Apply time masking to spectrogram.
        
        Args:
            spec: Mel-spectrogram tensor (channels, freq, time)
            
        Returns:
            Masked spectrogram
        """
        spec = spec.clone()
        time_len = spec.shape[2]
        mask_len = np.random.randint(0, self.time_mask_param)
        mask_start = np.random.randint(0, max(1, time_len - mask_len))
        spec[:, :, mask_start:mask_start + mask_len] = 0
        return spec
    
    def freq_mask(self, spec):
        """
        Apply frequency masking to spectrogram.
        
        Args:
            spec: Mel-spectrogram tensor (channels, freq, time)
            
        Returns:
            Masked spectrogram
        """
        spec = spec.clone()
        freq_len = spec.shape[1]
        mask_len = np.random.randint(0, self.freq_mask_param)
        mask_start = np.random.randint(0, max(1, freq_len - mask_len))
        spec[:, mask_start:mask_start + mask_len, :] = 0
        return spec
    
    def add_noise(self, spec):
        """
        Add Gaussian noise to spectrogram.
        
        Args:
            spec: Mel-spectrogram tensor
            
        Returns:
            Noisy spectrogram
        """
        noise = torch.randn_like(spec) * self.noise_factor
        return spec + noise
    
    def time_shift(self, spec, shift_max=20):
        """
        Shift spectrogram in time dimension.
        
        Args:
            spec: Mel-spectrogram tensor
            shift_max: Maximum shift in time steps
            
        Returns:
            Shifted spectrogram
        """
        spec = spec.clone()
        shift = np.random.randint(-shift_max, shift_max)
        return torch.roll(spec, shift, dims=2)
    
    def pitch_shift(self, audio, sr=22050, n_steps=2):
        """
        Shift pitch of audio (applied before mel-spectrogram conversion).
        
        Args:
            audio: Audio time series
            sr: Sample rate
            n_steps: Number of steps to shift (can be negative)
            
        Returns:
            Pitch-shifted audio
        """
        n_steps = np.random.randint(-n_steps, n_steps + 1)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def time_stretch(self, audio, rate_range=(0.8, 1.2)):
        """
        Time-stretch audio (applied before mel-spectrogram conversion).
        
        Args:
            audio: Audio time series
            rate_range: Range of stretching rates
            
        Returns:
            Time-stretched audio
        """
        rate = np.random.uniform(rate_range[0], rate_range[1])
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def __call__(self, spec, apply_prob=0.5):
        """
        Apply random augmentations to spectrogram.
        
        Args:
            spec: Input mel-spectrogram tensor
            apply_prob: Probability of applying each augmentation
            
        Returns:
            Augmented spectrogram
        """
        if np.random.random() < apply_prob:
            spec = self.time_mask(spec)
        if np.random.random() < apply_prob:
            spec = self.freq_mask(spec)
        if np.random.random() < apply_prob:
            spec = self.add_noise(spec)
        if np.random.random() < apply_prob:
            spec = self.time_shift(spec)
            
        return spec
    
    def create_pair(self, spec):
        """
        Create a pair of augmented views for contrastive learning.
        
        Args:
            spec: Input mel-spectrogram tensor
            
        Returns:
            Tuple of two augmented versions
        """
        view1 = self(spec.clone())
        view2 = self(spec.clone())
        return view1, view2
