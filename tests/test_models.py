"""
Basic tests for MarineView Project components.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import SimCLR, AcousticCNN, RareSpeciesClassifier
from src.models.simclr import NT_Xent
from src.losses import FocalLoss, ClassBalancedLoss
from src.data import AcousticDataset, AudioPreprocessor, AcousticAugmentation


def test_acoustic_cnn():
    """Test AcousticCNN forward pass"""
    print("Testing AcousticCNN...")
    model = AcousticCNN(input_channels=1, num_features=512)
    x = torch.randn(2, 1, 128, 128)
    output = model(x)
    assert output.shape == (2, 512), f"Expected shape (2, 512), got {output.shape}"
    print("✓ AcousticCNN test passed")


def test_simclr():
    """Test SimCLR model"""
    print("Testing SimCLR...")
    encoder = AcousticCNN(input_channels=1, num_features=512)
    model = SimCLR(base_encoder=encoder, projection_dim=128, hidden_dim=512)
    x = torch.randn(2, 1, 128, 128)
    features, projections = model(x)
    assert features.shape == (2, 512), f"Expected features shape (2, 512), got {features.shape}"
    assert projections.shape == (2, 128), f"Expected projections shape (2, 128), got {projections.shape}"
    print("✓ SimCLR test passed")


def test_nt_xent_loss():
    """Test NT-Xent loss"""
    print("Testing NT-Xent loss...")
    criterion = NT_Xent(temperature=0.5, batch_size=4)
    z_i = torch.randn(4, 128)
    z_j = torch.randn(4, 128)
    loss = criterion(z_i, z_j)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ NT-Xent loss test passed (loss: {loss.item():.4f})")


def test_focal_loss():
    """Test Focal Loss"""
    print("Testing Focal Loss...")
    criterion = FocalLoss(alpha=None, gamma=2.0)
    inputs = torch.randn(4, 5)  # 4 samples, 5 classes
    targets = torch.tensor([0, 1, 2, 3])
    loss = criterion(inputs, targets)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Focal Loss test passed (loss: {loss.item():.4f})")


def test_class_balanced_loss():
    """Test Class-Balanced Loss"""
    print("Testing Class-Balanced Loss...")
    samples_per_class = [100, 50, 30, 20, 10]
    criterion = ClassBalancedLoss(samples_per_class=samples_per_class, beta=0.9999)
    inputs = torch.randn(4, 5)
    targets = torch.tensor([0, 1, 4, 4])
    loss = criterion(inputs, targets)
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Class-Balanced Loss test passed (loss: {loss.item():.4f})")


def test_classifier():
    """Test RareSpeciesClassifier"""
    print("Testing RareSpeciesClassifier...")
    encoder = AcousticCNN(input_channels=1, num_features=512)
    model = RareSpeciesClassifier(encoder=encoder, num_classes=5, feature_dim=512)
    x = torch.randn(2, 1, 128, 128)
    output = model(x)
    assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"
    print("✓ RareSpeciesClassifier test passed")


def test_audio_preprocessor():
    """Test AudioPreprocessor"""
    print("Testing AudioPreprocessor...")
    preprocessor = AudioPreprocessor(sr=22050, n_mels=128)
    
    # Test with dummy audio
    audio = np.random.randn(22050 * 5)  # 5 seconds of audio
    tensor = preprocessor.preprocess_array(audio)
    assert tensor.shape[0] == 1, "Should have 1 channel"
    assert tensor.shape[1] == 128, "Should have 128 mel bands"
    print(f"✓ AudioPreprocessor test passed (output shape: {tensor.shape})")


def test_augmentation():
    """Test AcousticAugmentation"""
    print("Testing AcousticAugmentation...")
    augmentation = AcousticAugmentation(time_mask_param=20, freq_mask_param=20)
    spec = torch.randn(1, 128, 128)
    
    # Test single augmentation
    aug_spec = augmentation(spec)
    assert aug_spec.shape == spec.shape, "Augmented spec should have same shape"
    
    # Test pair creation
    view1, view2 = augmentation.create_pair(spec)
    assert view1.shape == spec.shape, "View1 should have same shape"
    assert view2.shape == spec.shape, "View2 should have same shape"
    print("✓ AcousticAugmentation test passed")


def test_dataset():
    """Test AcousticDataset"""
    print("Testing AcousticDataset...")
    
    # Create dummy data
    data = [torch.randn(1, 128, 128) for _ in range(10)]
    labels = [i % 5 for i in range(10)]
    
    # Test regular dataset
    dataset = AcousticDataset(data, labels, mode='train')
    spec, label = dataset[0]
    assert spec.shape[0] == 1, "Should have 1 channel"
    assert isinstance(label, int), "Label should be integer"
    
    # Test with pairs
    dataset_pairs = AcousticDataset(data, labels, mode='train', return_pairs=True)
    (view1, view2), label = dataset_pairs[0]
    assert view1.shape == view2.shape, "Paired views should have same shape"
    print("✓ AcousticDataset test passed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("Running MarineView Project Tests")
    print("="*60 + "\n")
    
    try:
        test_acoustic_cnn()
        test_simclr()
        test_nt_xent_loss()
        test_focal_loss()
        test_class_balanced_loss()
        test_classifier()
        test_audio_preprocessor()
        test_augmentation()
        test_dataset()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
