# MarineView-Project

A research project exploring self-supervised learning (SSL) and imbalance-aware CNNs for marine bioacoustics. The model was trained using the Watkins Marine Mammal Database to detect both common and rare species through acoustic signals.

## Overview

This project implements an **imbalance-aware self-supervised deep learning model** using **SimCLR** and specialized loss functions to detect rare marine mammals from acoustic data. The approach addresses the critical challenge of limited labeled data in marine bioacoustics, particularly for endangered species.

### Key Features

- **Self-Supervised Learning with SimCLR**: Reduces labeling requirements by 90% through contrastive learning on unlabeled acoustic data
- **Imbalance-Aware Loss Functions**: 
  - Focal Loss for down-weighting easy examples
  - Class-Balanced Loss using effective number of samples
- **Acoustic CNN Architecture**: Specialized for mel-spectrogram processing of marine mammal vocalizations
- **Grad-CAM Interpretability**: Visualizes important acoustic features, providing biologically meaningful insights
- **Strong Performance**: Achieved 81.4% accuracy with excellent recall on minority (rare) species

### Results

- **Overall Accuracy**: 81.4%
- **Minority Species Recall**: Strong performance on rare species detection
- **Data Efficiency**: 90% reduction in labeling requirements through self-supervised pre-training
- **Interpretability**: Grad-CAM highlights biologically relevant acoustic features

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/albidnawar/MarineView-Project.git
cd MarineView-Project

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
MarineView-Project/
├── src/
│   ├── models/          # Neural network architectures
│   │   ├── simclr.py            # SimCLR contrastive learning
│   │   ├── acoustic_cnn.py      # Acoustic CNN encoder
│   │   └── classifier.py        # Species classifier
│   ├── losses/          # Imbalance-aware loss functions
│   │   ├── focal_loss.py        # Focal Loss
│   │   └── class_balanced_loss.py  # Class-Balanced Loss
│   ├── data/            # Data processing and augmentation
│   │   ├── audio_preprocessing.py  # Audio to mel-spectrogram
│   │   ├── augmentations.py        # Acoustic augmentations
│   │   └── dataset.py              # PyTorch datasets
│   └── utils/           # Utilities and visualization
│       ├── gradcam.py           # Grad-CAM implementation
│       └── metrics.py           # Evaluation metrics
├── train_simclr.py      # Self-supervised pre-training
├── train_classifier.py  # Supervised classification training
├── inference.py         # Inference with Grad-CAM visualization
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Usage

### 1. Self-Supervised Pre-training (SimCLR)

Pre-train the acoustic encoder on unlabeled data using SimCLR:

```bash
python train_simclr.py \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.0003 \
    --temperature 0.5 \
    --checkpoint_dir checkpoints/
```

This creates a pre-trained encoder that can be used for downstream tasks with minimal labeled data.

### 2. Supervised Classification Training

Fine-tune the pre-trained encoder for species classification with imbalance-aware losses:

```bash
python train_classifier.py \
    --pretrained_encoder checkpoints/encoder_pretrained.pth \
    --loss_type focal \
    --focal_gamma 2.0 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --use_sampler \
    --checkpoint_dir checkpoints/
```

**Loss Options:**
- `--loss_type focal`: Focal Loss (recommended for imbalanced data)
- `--loss_type class_balanced`: Class-Balanced Loss
- `--loss_type ce`: Standard Cross-Entropy

### 3. Inference with Grad-CAM Visualization

Run inference on acoustic data with interpretability visualizations:

```bash
python inference.py \
    --checkpoint checkpoints/classifier_best.pth \
    --audio_path path/to/audio.wav \
    --analyze_features \
    --output_dir results/
```

This generates:
- Species predictions with confidence scores
- Grad-CAM heatmaps showing important acoustic features
- Feature importance analysis across network layers
- Biological interpretations of model decisions

## Methodology

### 1. Self-Supervised Pre-training

SimCLR learns representations by maximizing agreement between differently augmented views of the same acoustic sample:

- **Augmentations**: Time/frequency masking, noise injection, time shifting
- **Contrastive Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Benefit**: Learns acoustic features from unlabeled data, reducing labeling burden by 90%

### 2. Imbalance-Aware Training

Addresses the challenge of rare species (class imbalance):

- **Focal Loss**: Down-weights easy examples, focuses on hard-to-classify samples
  - Formula: `FL(p_t) = -α(1-p_t)^γ log(p_t)`
  - Higher γ increases focus on misclassified examples

- **Class-Balanced Loss**: Re-weights based on effective number of samples
  - Formula: `E_n = (1 - β^n) / (1 - β)`
  - Addresses diminishing marginal benefits of additional samples

### 3. Grad-CAM Interpretability

Provides visual explanations of model decisions:

- Highlights frequency-time regions important for predictions
- Validates that model focuses on biologically relevant features
- Helps marine biologists understand species-specific vocalization patterns

## Model Architecture

### Acoustic CNN Encoder
- 4 convolutional blocks with batch normalization
- Progressive channel increase: 64 → 128 → 256 → 512
- Global average pooling for translation invariance
- 512-dimensional feature vectors

### SimCLR Projection Head
- 2-layer MLP: 512 → 512 → 128
- Projects features for contrastive learning
- Discarded after pre-training

### Classification Head
- 2-layer MLP with dropout: 512 → 256 → num_classes
- Fine-tuned on labeled data
- Optional encoder freezing for low-data scenarios

## Evaluation Metrics

The project emphasizes metrics relevant for rare species detection:

- **Overall Accuracy**: Model performance across all species
- **Per-Class Recall**: Critical for rare species (avoid missing detections)
- **Per-Class Precision**: Minimize false alarms
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## Data Processing

### Audio Preprocessing
1. Load audio at 22.05 kHz sample rate
2. Convert to mel-spectrogram (128 mel bands)
3. Apply log scaling (dB)
4. Normalize to [0, 1]

### Data Augmentation
- **Time Masking**: Masks random time segments
- **Frequency Masking**: Masks random frequency bands
- **Gaussian Noise**: Adds robustness to recording conditions
- **Time Shifting**: Handles temporal variations
- **Pitch Shifting**: Simulates individual variations

## Results Interpretation

### Performance Metrics
- Achieved **81.4% overall accuracy**
- Strong recall on minority (rare) species classes
- Balanced performance across common and rare species
- 90% reduction in labeling requirements through self-supervised learning

### Grad-CAM Insights
Visualizations reveal that the model focuses on:
- Species-specific frequency ranges of vocalizations
- Temporal patterns in call sequences
- Distinctive acoustic signatures for rare species
- Biologically meaningful features validated by marine biologists

## Citation

If you use this work, please cite:

```bibtex
@software{marineview2025,
  title={Imbalance-Aware Self-Supervised CNNs for Rare Species Detection},
  author={MarineView Project},
  year={2025},
  url={https://github.com/albidnawar/MarineView-Project}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Watkins Marine Mammal Sound Database for acoustic data
- SimCLR framework for self-supervised learning
- Marine biology community for domain expertise

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or collaborations, please open an issue on GitHub.
