"""
Grad-CAM implementation for interpretability of acoustic CNN models.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN interpretability.
    
    Grad-CAM produces visual explanations by highlighting the regions of the 
    spectrogram that are important for predictions, providing biologically 
    meaningful insights into which acoustic features the model focuses on.
    
    Args:
        model: The neural network model
        target_layer: The layer to compute gradients for (typically last conv layer)
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the input.
        
        Args:
            input_tensor: Input spectrogram tensor (1, C, H, W)
            target_class: Target class for CAM (if None, uses predicted class)
            
        Returns:
            cam: Class activation map (H, W)
            prediction: Model prediction (class index)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get prediction if target_class not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), target_class
    
    def __call__(self, input_tensor, target_class=None):
        """Convenience method"""
        return self.generate_cam(input_tensor, target_class)


def visualize_gradcam(original_spec, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Visualize Grad-CAM heatmap overlaid on original spectrogram.
    
    Args:
        original_spec: Original spectrogram (H, W) or (C, H, W)
        cam: Class activation map (H, W)
        alpha: Transparency of overlay
        colormap: OpenCV colormap for heatmap
        
    Returns:
        visualization: RGB visualization of Grad-CAM
    """
    # Handle channel dimension
    if isinstance(original_spec, torch.Tensor):
        if original_spec.dim() == 3:
            original_spec = original_spec[0]  # Remove channel dimension
        original_spec = original_spec.cpu().numpy()
    
    # Resize CAM to match spectrogram size
    cam_resized = cv2.resize(cam, (original_spec.shape[1], original_spec.shape[0]))
    
    # Normalize spectrogram to [0, 255]
    spec_normalized = ((original_spec - original_spec.min()) / 
                       (original_spec.max() - original_spec.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert to RGB
    spec_rgb = cv2.cvtColor(spec_normalized, cv2.COLOR_GRAY2RGB)
    
    # Apply colormap to CAM
    cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
    
    # Overlay
    visualization = cv2.addWeighted(spec_rgb, 1 - alpha, cam_colored, alpha, 0)
    
    return visualization


def plot_gradcam_analysis(model, input_tensor, target_layer, class_names=None, 
                          save_path=None):
    """
    Create a comprehensive Grad-CAM analysis plot.
    
    Args:
        model: Neural network model
        input_tensor: Input spectrogram tensor
        target_layer: Target layer for Grad-CAM
        class_names: List of class names for labeling
        save_path: Path to save the figure
    """
    gradcam = GradCAM(model, target_layer)
    cam, pred_class = gradcam(input_tensor)
    
    # Get original spectrogram
    if input_tensor.dim() == 4:
        spec = input_tensor[0]
    else:
        spec = input_tensor
    
    # Create visualization
    vis = visualize_gradcam(spec, cam)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original spectrogram
    axes[0].imshow(spec[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original Mel-Spectrogram')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Frequency (Mel)')
    
    # Heatmap
    axes[1].imshow(cam, aspect='auto', origin='lower', cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency (Mel)')
    
    # Overlay
    axes[2].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Grad-CAM Overlay')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Frequency (Mel)')
    
    # Add prediction label
    pred_label = class_names[pred_class] if class_names else f"Class {pred_class}"
    fig.suptitle(f'Grad-CAM Analysis - Predicted: {pred_label}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
