
# Fish Image Classification Project

##  Project Overview

This project builds a **Fish Species Classifier** using a **custom Convolutional Neural Network (CNN)** from scratch (without using any pretrained models).  
The network is trained on a dataset of fish images, organized by species classes, to recognize and classify them accurately.

##  Model Architecture

- **Input Layer**: Accepts RGB images resized to 128x128
- **Conv Block 1**: Conv2D (32 filters) → ReLU → MaxPooling
- **Conv Block 2**: Conv2D (64 filters) → ReLU → MaxPooling
- **Fully Connected Block**: Flatten → Dense (512) → ReLU → Dense (num_classes)
- 
## Project Structure
```
.
├── config.py            # Main configuration and execution script
├── dataset.py           # Dataset loading and preprocessing
├── model.py            # CNN model definition
├── predict.py          # Inference functions
└── train.py            # Training loop implementation
```
## Training details:
-Optimizer: Adam with learning rate 0.001
-Loss function: CrossEntropyLoss with label smoothing (0.1)
-Batch size: 32
-Image size: 128x128
-Data augmentation: Random horizontal flips and rotations

## Requirements: 
-Python 3.x
-PyTorch
-torchvision
-scikit-learn
-matplotlib

## Install dependencies:
1. Clone this repository.
2. Install the required packages:
```bash
pip install torch torchvision matplotlib scikit-learn
```

##  Dataset Structure

Organize your dataset like:
```
FishImgDataset/
├── train/
│   ├── salmon/
│   ├── tuna/
│   └── trout/
└── val/
    ├── salmon/
    ├── tuna/
    └── trout/
```

##  How to Use

Train the model:
```bash
python main.py
```

This will:
- Train and validate
- Save best model to `/checkpoints/best_model.pth`
- Plot curves
- Show confusion matrix

##  Results

- Training/Validation loss and accuracy plots
- Confusion matrix for validation
- Best model automatically saved

##  Author

Built and organized by Sumit Singh(20221269).
