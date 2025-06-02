# data loader for pytorch

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, csv_dir, transform=None):
        self.csv_dir = csv_dir
        self.transform = transform
        
    # Load image paths and labels from the CSV file
    def load_data(self):
        import pandas as pd
        df = pd.read_csv(self.csv_dir)
        self.image_paths = df['image_path'].tolist()
        self.labels = df['label'].tolist()
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.labels = [self.class_to_idx[label] for label in self.labels]
        return self.image_paths, self.labels, self.class_to_idx
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# debug
if __name__ == '__main__':
    csv_dir = '/Users/iyongjeong/WORK/dacon/clf/data/train_csv/train_file_list.csv'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(csv_dir=csv_dir, transform=transform)
    dataset.load_data()
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break