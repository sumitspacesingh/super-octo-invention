
from model import MyCustomModel as TheModel
from train import my_descriptively_named_train_function as the_trainer
from predict import cryptic_inf_f as the_predictor
from dataset import UnicornImgDataset as TheDataset
from dataset import unicornLoader as the_dataloader

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Datasets
root_dir = '/content/drive/MyDrive/FishImgDataset'
train_data = TheDataset(root_dir, train=True).get_dataset()
val_data = TheDataset(root_dir, train=False).get_dataset()

train_loader = the_dataloader(train_data, batch_size=32, shuffle=True)
val_loader = the_dataloader(val_data, batch_size=32, shuffle=False)

# Load Model
model = TheModel(num_classes=len(train_data.classes)).to(device)

# Train
train_losses, val_losses, train_accs, val_accs, all_preds, all_labels = the_trainer(
    model, train_loader, val_loader, device, num_epochs=15
)

# Plot Loss and Accuracy
epochs = range(1, 16)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_accs, label='Train Accuracy')
plt.plot(epochs, val_accs, label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_data.classes)
disp.plot(xticks_rotation=45, cmap="Blues")
plt.title("Confusion Matrix on Validation Set")
plt.show()
