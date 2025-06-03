import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from functools import lru_cache
import pandas as pd
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, csv_dir, transform=None):
        # CSV 한 번만 읽기
        self.df = pd.read_csv(csv_dir)
        self.image_paths = self.df['image_path'].tolist()
        self.labels = self.df['label'].tolist()
        # 클래스 매핑
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @lru_cache(maxsize=64)
    def load_image(self, img_path):
        return Image.open(img_path).convert('RGB')  # PIL 사용

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = self.load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
# debug
if __name__ == '__main__':
    csv_dir = r'C:\works\dacon\img_clf\data\train_csv\train_file_list_100.csv'
    
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