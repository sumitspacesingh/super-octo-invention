
# Fish Image Classification Project

##  Project Overview

This project builds a **Fish Species Classifier** using a **custom Convolutional Neural Network (CNN)** from scratch (without using any pretrained models).  
The network is trained on a dataset of fish images, organized by species classes, to recognize and classify them accurately.

✅ Modular code structure (model, training, prediction, dataset handling)   
✅ Clean results with plotted training curves and confusion matrix

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

##  Installation
For The Data (10 images per class from test set) could be installed from google drive link below:
https://drive.google.com/drive/folders/1skVk-03lETlwXrmM3KH9u5fL8zw7K13b?usp=sharing
(Because the size of this file is more than 25 MB.
Clone this repository:
```bash
git clone https://github.com/your-username/FishClassificationProject.git
cd FishClassificationProject
```

Install dependencies:
```bash
pip install torch torchvision matplotlib scikit-learn
```

## 🗂️ Dataset Structure

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

Upload it to `/MyDrive/FishImgDataset/` if using Google Drive.

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
- Due to larger file Git was unable to upload the final weight file, so here is the link to google drive where you can find it:
  https://drive.google.com/file/d/10n6VG9hOQ-5xbFTib6t8uM82WoIv5vxx/view?usp=sharing
  (Note- It might show a problem in viewing due to it's large size, so download it)

##  Author

Built and organized by Sumit Singh(20221269).
