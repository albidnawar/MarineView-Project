"""
Training script for supervised classification with imbalance-aware loss.

This script fine-tunes a pre-trained encoder (from SimCLR) for rare species
classification using imbalance-aware loss functions.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
from tqdm import tqdm
import os
import numpy as np

from src.models import AcousticCNN, RareSpeciesClassifier
from src.losses import FocalLoss, ClassBalancedLoss
from src.data import AcousticDataset, AudioPreprocessor, AcousticAugmentation
from src.utils import compute_metrics, print_metrics, plot_confusion_matrix


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics, all_preds, all_labels


def main(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data preprocessor and augmentation
    preprocessor = AudioPreprocessor(
        sr=args.sample_rate,
        n_mels=args.n_mels,
        duration=args.duration
    )
    
    augmentation = AcousticAugmentation(
        time_mask_param=args.time_mask,
        freq_mask_param=args.freq_mask
    )
    
    # Create datasets
    print("Creating datasets...")
    print("Note: This is a demo. Replace with actual audio file paths and labels.")
    
    # Create dummy imbalanced dataset for demonstration
    # Simulating rare species scenario
    num_samples = 500
    num_classes = 5
    
    # Create imbalanced distribution (some rare species)
    class_samples = [200, 150, 100, 30, 20]  # Last two are "rare"
    dummy_data = []
    dummy_labels = []
    
    for class_id, n_samples in enumerate(class_samples):
        dummy_data.extend([torch.randn(1, 128, 128) for _ in range(n_samples)])
        dummy_labels.extend([class_id] * n_samples)
    
    # Split into train and validation
    split_idx = int(0.8 * len(dummy_data))
    indices = np.random.permutation(len(dummy_data))
    
    train_data = [dummy_data[i] for i in indices[:split_idx]]
    train_labels = [dummy_labels[i] for i in indices[:split_idx]]
    val_data = [dummy_data[i] for i in indices[split_idx:]]
    val_labels = [dummy_labels[i] for i in indices[split_idx:]]
    
    train_dataset = AcousticDataset(
        data_paths=train_data,
        labels=train_labels,
        augmentation=augmentation,
        mode='train'
    )
    
    val_dataset = AcousticDataset(
        data_paths=val_data,
        labels=val_labels,
        mode='val'
    )
    
    # Create data loaders
    if args.use_sampler:
        # Use weighted sampler for imbalanced data
        from src.data.dataset import ImbalancedSampler
        sampler = ImbalancedSampler(train_labels, oversample_factor=2.0)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=WeightedRandomSampler(sampler.get_weights(), len(train_dataset)),
            num_workers=args.num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating classification model...")
    encoder = AcousticCNN(input_channels=1, num_features=512)
    
    # Load pre-trained encoder if available
    if args.pretrained_encoder:
        print(f"Loading pre-trained encoder from {args.pretrained_encoder}")
        encoder.load_state_dict(torch.load(args.pretrained_encoder))
    
    model = RareSpeciesClassifier(
        encoder=encoder,
        num_classes=num_classes,
        feature_dim=512,
        freeze_encoder=args.freeze_encoder
    ).to(device)
    
    # Loss function - choose imbalance-aware loss
    if args.loss_type == 'focal':
        # Calculate class weights
        unique, counts = np.unique(train_labels, return_counts=True)
        alpha = 1.0 / counts
        alpha = alpha / alpha.sum()
        criterion = FocalLoss(alpha=alpha.tolist(), gamma=args.focal_gamma)
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    elif args.loss_type == 'class_balanced':
        unique, counts = np.unique(train_labels, return_counts=True)
        criterion = ClassBalancedLoss(
            samples_per_class=counts.tolist(),
            beta=args.cb_beta,
            loss_type='focal'
        )
        print(f"Using Class-Balanced Loss (beta={args.cb_beta})")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Using standard Cross-Entropy Loss")
    
    criterion = criterion.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validate
        val_loss, val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
                'metrics': val_metrics
            }
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'classifier_best.pth'))
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'classifier_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_metrics, final_preds, final_labels = validate(model, val_loader, criterion, device)
    
    class_names = [f"Species {i}" for i in range(num_classes)]
    print_metrics(final_metrics, class_names)
    
    # Plot confusion matrix
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    os.makedirs('results', exist_ok=True)
    plot_confusion_matrix(
        final_metrics['confusion_matrix'],
        class_names=class_names,
        save_path='results/confusion_matrix.png'
    )
    print("\nConfusion matrix saved to results/confusion_matrix.png")
    
    # Analyze minority species performance
    from src.utils.metrics import analyze_minority_performance
    minority_metrics = analyze_minority_performance(
        final_metrics,
        class_samples,
        class_names,
        minority_threshold=0.15
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Overall Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Recall on Minority Species: {minority_metrics['minority_avg_recall']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rare Species Classification Training")
    
    # Data parameters
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--time_mask', type=int, default=20)
    parser.add_argument('--freq_mask', type=int, default=20)
    
    # Model parameters
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                       help='Path to pre-trained encoder')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder during training')
    
    # Loss parameters
    parser.add_argument('--loss_type', type=str, default='focal',
                       choices=['focal', 'class_balanced', 'ce'],
                       help='Type of loss function')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma for focal loss')
    parser.add_argument('--cb_beta', type=float, default=0.9999,
                       help='Beta for class-balanced loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_sampler', action='store_true',
                       help='Use weighted sampler for imbalanced data')
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    main(args)
