"""
Audio preprocessing for marine mammal acoustic signals.
"""

import numpy as np
import librosa
import torch


class AudioPreprocessor:
    """
    Preprocessor for converting raw audio to mel-spectrograms.
    
    Marine mammal vocalizations span various frequency ranges, so proper
    preprocessing is crucial for effective feature extraction.
    
    Args:
        sr (int): Sample rate for audio
        n_mels (int): Number of mel frequency bins
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
        duration (float): Duration of audio clips in seconds
    """
    
    def __init__(self, sr=22050, n_mels=128, n_fft=2048, hop_length=512, duration=5.0):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.target_length = int(sr * duration)
        
    def load_audio(self, audio_path):
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            audio: Audio time series
        """
        audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        # Pad or truncate to target length
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.target_length]
            
        return audio
    
    def audio_to_melspectrogram(self, audio):
        """
        Convert audio to mel-spectrogram.
        
        Args:
            audio: Audio time series
            
        Returns:
            mel_spec: Mel-spectrogram (n_mels, time_steps)
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def normalize(self, mel_spec):
        """
        Normalize mel-spectrogram to [0, 1] range.
        
        Args:
            mel_spec: Mel-spectrogram
            
        Returns:
            normalized: Normalized mel-spectrogram
        """
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        return mel_spec
    
    def preprocess(self, audio_path):
        """
        Complete preprocessing pipeline.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            tensor: Preprocessed mel-spectrogram as PyTorch tensor
        """
        audio = self.load_audio(audio_path)
        mel_spec = self.audio_to_melspectrogram(audio)
        mel_spec = self.normalize(mel_spec)
        
        # Convert to tensor and add channel dimension
        tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
        
        return tensor
    
    def preprocess_array(self, audio):
        """
        Preprocess audio array directly.
        
        Args:
            audio: Audio time series array
            
        Returns:
            tensor: Preprocessed mel-spectrogram as PyTorch tensor
        """
        mel_spec = self.audio_to_melspectrogram(audio)
        mel_spec = self.normalize(mel_spec)
        
        # Convert to tensor and add channel dimension
        tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
        
        return tensor
