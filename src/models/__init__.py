"""
Models for self-supervised learning and classification.
"""

from .simclr import SimCLR
from .acoustic_cnn import AcousticCNN
from .classifier import RareSpeciesClassifier

__all__ = ['SimCLR', 'AcousticCNN', 'RareSpeciesClassifier']
