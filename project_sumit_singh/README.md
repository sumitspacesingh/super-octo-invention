
# Fish Image Classification Project

##  Project Overview

This project builds a **Fish Species Classifier** using a **custom Convolutional Neural Network (CNN)** from scratch (without using any pretrained models).  
The network is trained on a dataset of fish images, organized by species classes, to recognize and classify them accurately.

##  Model Architecture

- **Input Layer**: Accepts RGB images resized to 128x128
- **Conv Block 1**: Conv2D (32 filters) → ReLU → MaxPooling
- **Conv Block 2**: Conv2D (64 filters) → ReLU → MaxPooling
- **Fully Connected Block**: Flatten → Dense (512) → ReLU → Dense (num_classes)

Training details:
- **Loss**: CrossEntropyLoss (with label smoothing)
- **Optimizer**: Adam (lr=0.001)
- **Augmentation**: Random rotation, horizontal flip, normalization
- **Epochs**: 15


Install dependencies:
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

##  Predict

After training, use the predictor:
```python
from predict import cryptic_inf_f as the_predictor
predicted_class_idx = the_predictor(model, preprocessed_image_tensor, device)
```

##  Results

- Training/Validation loss and accuracy plots
- Confusion matrix for validation
- Best model automatically saved

##  Author

Built and organized by Sumit Singh(20221269).
