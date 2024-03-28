import os 
from torch.utils.data import Dataset
class ImageDataset(Dataset):
    def __init__(self, labels, imgs, transform=None, target_transform=None):
        self.img_labels = labels
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label