
# Fish Image Classification Project

##  Project Overview

This project builds a **Fish Species Classifier** using a **custom Convolutional Neural Network (CNN)** from scratch (without using any pretrained models).  
The network is trained on a dataset of fish images, organized by species classes, to recognize and classify them accurately.
## Problem Statement: 
To identify types of fresh water fishes found which could help
regulate the fishing rate to help maintain a survivable ratio between these respective
fishes, to have an ecological balance. Input will be pictures of different species of
fresh water fishes namely: &#39;Bangus&#39;, &#39;Big Head Carp&#39;, &#39;Black Spotted Barb&#39;, &#39;Catfish&#39;,
&#39;Climbing Perch&#39;, &#39;Fourfinger Threadfin&#39;, &#39;Freshwater Eel&#39;, &#39;Glass Perchlet&#39;, &#39;Goby&#39;,
&#39;Gold Fish&#39;, &#39;Gourami&#39;, &#39;Grass Carp&#39;, &#39;Green Spotted Puffer&#39;, &#39;Indian Carp&#39;, &#39;Indo-Pacific
Tarpon&#39;, &#39;Jaguar Gapote&#39;, &#39;Janitor Fish&#39;, &#39;Knifefish&#39;, &#39;Long-Snouted Pipefish&#39;, &#39;Mosquito
Fish&#39;, &#39;Mudfish&#39;, &#39;Mullet&#39;, &#39;Pangasius&#39;, &#39;Perch&#39;, &#39;Scat Fish&#39;, &#39;Silver Barb&#39;, &#39;Silver
Carp&#39;,&#39;Silver Perch&#39;, &#39;Snakehead&#39;, &#39;Tenpounder&#39;, and &#39;Tilapia&#39;. 
- Output will be the name
of the species picture provided.
- Data Source I will be using is taken from Kaggle fish dataset. More specifically, fish
species that can be found at the Marine Fishing Port in Cabuyao City.
- For training set
I will be using 8791 images distributed among the above-named species.
- For testing
we will have 2751 images. We are choosing this model because this model has a
lot of different fish species in a small area which increases interactions between
these fishes, which in turn making this monitoring of species important.

##  Model Architecture

- **Input Layer**: Accepts RGB images resized to 128x128
- **Conv Block 1**: Conv2D (32 filters) → ReLU → MaxPooling
- **Conv Block 2**: Conv2D (64 filters) → ReLU → MaxPooling
- **Fully Connected Block**: Flatten → Dense (512) → ReLU → Dense (num_classes)
  
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
- Optimizer: Adam with learning rate 0.001
- Loss function: CrossEntropyLoss with label smoothing (0.1)
- Batch size: 32
- Image size: 128x128
- Data augmentation: Random horizontal flips and rotations

## Requirements: 
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib

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
