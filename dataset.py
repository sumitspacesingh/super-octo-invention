
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class UnicornImgDataset:
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            self.dataset = datasets.ImageFolder(root=os.path.join(root_dir, 'train'), transform=self.transform)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            self.dataset = datasets.ImageFolder(root=os.path.join(root_dir, 'val'), transform=self.transform)

    def get_dataset(self):
        return self.dataset

def unicornLoader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
