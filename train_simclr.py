"""
Training script for SimCLR self-supervised pre-training.

This script demonstrates how to pre-train the acoustic CNN encoder using
SimCLR on unlabeled acoustic data, reducing the need for labeled samples.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

from src.models import SimCLR, AcousticCNN
from src.models.simclr import NT_Xent
from src.data import AcousticDataset, AudioPreprocessor, AcousticAugmentation


def train_simclr(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train SimCLR for one epoch.
    
    Args:
        model: SimCLR model
        dataloader: DataLoader with pairs of augmented views
        criterion: NT-Xent loss
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for (view1, view2), _ in pbar:
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        # Forward pass
        _, z1 = model(view1)
        _, z2 = model(view2)
        
        # Compute loss
        loss = criterion(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


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
    
    # Create dataset
    # Note: In practice, load your actual data paths
    print("Creating dataset...")
    print("Note: This is a demo. Replace with actual audio file paths.")
    
    # Create dummy dataset for demonstration
    dummy_data = [torch.randn(1, 128, 128) for _ in range(100)]
    
    dataset = AcousticDataset(
        data_paths=dummy_data,
        labels=None,  # No labels needed for self-supervised learning
        preprocessor=None,  # Already preprocessed
        augmentation=augmentation,
        mode='train',
        return_pairs=True  # Return pairs for contrastive learning
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating SimCLR model...")
    encoder = AcousticCNN(input_channels=1, num_features=512)
    model = SimCLR(
        base_encoder=encoder,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Loss and optimizer
    criterion = NT_Xent(temperature=args.temperature, batch_size=args.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"\nStarting SimCLR pre-training for {args.epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        loss = train_simclr(model, dataloader, criterion, optimizer, device, epoch)
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}")
        
        # Save checkpoint
        if loss < best_loss:
            best_loss = loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'simclr_best.pth'))
            print(f"Saved best checkpoint (loss: {loss:.4f})")
    
    # Save final encoder
    encoder_path = os.path.join(args.checkpoint_dir, 'encoder_pretrained.pth')
    torch.save(encoder.state_dict(), encoder_path)
    print(f"\nPre-training complete! Encoder saved to {encoder_path}")
    print("This encoder can now be used for downstream classification with reduced labeled data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Self-Supervised Pre-training")
    
    # Data parameters
    parser.add_argument('--sample_rate', type=int, default=22050, help='Audio sample rate')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of mel bands')
    parser.add_argument('--duration', type=float, default=5.0, help='Audio duration in seconds')
    parser.add_argument('--time_mask', type=int, default=20, help='Time masking parameter')
    parser.add_argument('--freq_mask', type=int, default=20, help='Frequency masking parameter')
    
    # Model parameters
    parser.add_argument('--projection_dim', type=int, default=128, help='Projection head output dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Projection head hidden dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for NT-Xent loss')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    main(args)
