"""
Evaluation metrics for rare species detection.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred, y_prob=None, average='weighted'):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        average: Averaging strategy for multi-class metrics
        
    Returns:
        metrics: Dictionary containing various metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Per-class metrics for minority species analysis
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics from compute_metrics
        class_names: List of class names for per-class metrics
    """
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    
    print("\n" + "="*60)
    print("PER-CLASS METRICS (Important for Rare Species)")
    print("="*60)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(metrics['precision_per_class']))]
    
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    for i, name in enumerate(class_names):
        print(f"{name:<20} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} "
              f"{metrics['f1_per_class'][i]:<12.4f}")


def plot_confusion_matrix(cm, class_names=None, normalize=False, 
                          figsize=(10, 8), save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        fig: Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_minority_performance(metrics, class_counts, class_names=None, 
                                 minority_threshold=0.1):
    """
    Analyze model performance on minority (rare) species.
    
    Args:
        metrics: Metrics dictionary from compute_metrics
        class_counts: Number of samples per class
        class_names: List of class names
        minority_threshold: Threshold ratio to consider a class as minority
        
    Returns:
        minority_metrics: Dictionary with minority class analysis
    """
    total_samples = sum(class_counts)
    class_ratios = np.array(class_counts) / total_samples
    
    # Identify minority classes
    minority_mask = class_ratios < minority_threshold
    majority_mask = ~minority_mask
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(class_counts))]
    
    minority_metrics = {
        'minority_classes': [class_names[i] for i in range(len(class_names)) if minority_mask[i]],
        'majority_classes': [class_names[i] for i in range(len(class_names)) if majority_mask[i]],
        'minority_avg_recall': np.mean(metrics['recall_per_class'][minority_mask]) if minority_mask.any() else 0.0,
        'majority_avg_recall': np.mean(metrics['recall_per_class'][majority_mask]) if majority_mask.any() else 0.0,
        'minority_avg_precision': np.mean(metrics['precision_per_class'][minority_mask]) if minority_mask.any() else 0.0,
        'majority_avg_precision': np.mean(metrics['precision_per_class'][majority_mask]) if majority_mask.any() else 0.0,
    }
    
    print("\n" + "="*60)
    print("MINORITY (RARE) SPECIES PERFORMANCE")
    print("="*60)
    print(f"Minority classes: {', '.join(minority_metrics['minority_classes'])}")
    print(f"Average Recall on Minority Species:    {minority_metrics['minority_avg_recall']:.4f}")
    print(f"Average Precision on Minority Species: {minority_metrics['minority_avg_precision']:.4f}")
    print(f"\nMajority classes: {', '.join(minority_metrics['majority_classes'])}")
    print(f"Average Recall on Majority Species:    {minority_metrics['majority_avg_recall']:.4f}")
    print(f"Average Precision on Majority Species: {minority_metrics['majority_avg_precision']:.4f}")
    
    return minority_metrics


def evaluate_model(model, dataloader, device='cuda', class_names=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Neural network model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        metrics: Evaluation metrics
        predictions: Model predictions
        labels: Ground truth labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    # Print metrics
    print_metrics(metrics, class_names)
    
    return metrics, all_preds, all_labels
