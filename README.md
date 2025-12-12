# MANE-4961-Final-Project
Final project for MANE 4961. Water droplet image segmentation U-net model
# Raindrop Detection on Windshield Images Using U-Net

**Author:** Peter Iascone  
**Course:** MANE 4962 - Machine Learning Applications  
**Date:** December 12, 2025

## Project Description

This project develops a deep learning model for automated detection and segmentation of water droplets on vehicle windshield images. The model produces binary masks identifying droplet locations, which can be used for autonomous vehicle vision systems to filter out obstructions from camera feeds.

## Objective

Build a U-Net convolutional neural network capable of producing pixel-level binary masks of water droplets from RGB input images, achieving high accuracy while maintaining low computational cost suitable for real-time applications.

## Dataset

The model was trained on the **Raindrops on Windshield** dataset by Soboleva and Shipitko:
- **Source:** [https://github.com/EvoCargo/RaindropsOnWindshield]
- **Total Images:** 8,190 images with corresponding binary masks
- **Content:** Images captured from vehicle-mounted cameras driving through urban environments

## Model Overview

### Architecture
- **Model Type:** U-Net (Encoder-Decoder with Skip Connections)
- **Encoder:** 4 layers with 32, 64, 128, 256 filters
- **Bottleneck:** 512 filters with 0.5 dropout
- **Decoder:** 4 layers mirroring encoder (256, 128, 64, 32 filters)
- **Output:** Sigmoid activation producing probability maps

### Training Configuration
- **Input Size:** 768 × 1024 pixels
- **Batch Size:** 6
- **Epochs:** 25
- **Learning Rate:** 8e-4 (with gradient clipping)
- **Loss Function:** Binary Cross-Entropy + Dice Loss
- **Optimizer:** Adam with clipvalue=0.2 and clipnorm=1.0

### Data Preprocessing
- **Class Balancing:** 30% of no-droplet images retained to address class imbalance
- **Train/Test Split:** 80/20
- **Data Augmentation:** Horizontal flip, brightness/contrast adjustment, Gaussian blur, gamma correction

## Results

| Metric | Value |
|--------|-------|
| **Validation Dice Coefficient** | 92.78% |
| **Pixel Accuracy** | 98.55% |
| **Training Time** | ~7 hours |

The model shows strong performance across all image subfolders, with best results on large, well-defined droplets and some challenges with small droplets and gray pavement backgrounds.
## Instructions to Run

### Prerequisites

1. **NVIDIA GPU** with CUDA support (required for training)
2. **CUDA Toolkit** (version 11.x or 12.x compatible with your TensorFlow version)
3. **cuDNN** (CUDA Deep Neural Network library, must match CUDA version)
4. **Python 3.10** (newest version compatible with CUDA/cuDNN at time of development)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/[repo-name].git
   cd [repo-name]
   ```

2. Install CUDA and cuDNN:
   - Download CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Download cuDNN from [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
   - Follow NVIDIA's installation instructions for your operating system

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   ```bash
   git clone https://github.com/EvoCargo/RaindropsOnWindshield.git
   ```

5. Update file paths in the scripts:
   - In `src/final_model_stable.py`, update `IMAGE_ROOT` and `MASK_ROOT` to point to your dataset location
   - In `src/Final_plots.py`, update `MODEL_PATH`, `HISTORY_PATH`, `IMAGE_ROOT`, and `MASK_ROOT`

### Training

```bash
python src/final_model_stable.py
```

The training script will:
- Automatically detect and configure GPU
- Apply class balancing and data augmentation
- Save checkpoints after each epoch
- Resume from the latest checkpoint if interrupted

### Evaluation

```bash
python src/Final_plots.py
```

This will generate:
- Training curve plots
- Performance metrics summary
- Sample predictions from each subfolder
- Detailed evaluation report

## Key Findings

1. **Class Imbalance:** Addressed by undersampling no-droplet images (keeping 30%)
2. **Numerical Stability:** Achieved through gradient clipping and reduced learning rate
3. **Generalization:** Model performs well on custom test images outside the training set
4. **Limitations:** Some false positives on gray pavement and monochromatic backgrounds

## References

1. Soboleva, V. and Shipitko, O. "Raindrops on Windshield: Dataset and Lightweight Gradient-Based Detection Algorithm." arXiv:2104.05078, 2021.

