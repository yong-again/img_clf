from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import sys
import torch

sys.path.append('../')  # Adjust the path as necessary
from base import BaseDataLoader
from utils.util import get_parent_path

class CarImageDataset(Dataset):
    """
    Car Image Classification dataset
    """
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.csv_file = os.path.join(data_dir, 'train_csv', csv_file)
        self.transform = transform
        self.data_frame = pd.read_csv(self.csv_file)
        self.num_classes = len(self.data_frame['label_index'].unique())
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_path = os.path.basename(self.data_frame.iloc[idx]['image_path'])
        label = self.data_frame.iloc[idx]['label_index']
        folder_name = self.data_frame.iloc[idx]['folder_name']
        img_full_path = f"{self.data_dir}/train/{folder_name}/{img_path}"
        
        image = Image.open(img_full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.data_frame.iloc[idx]['label_index']
        label_onehot = torch.eye(self.num_classes)[label]
        
        return image, label_onehot

class CarImageDataLoader(BaseDataLoader):
    """
    Car Image Classfication dataset
    """
    def __init__(self, data_dir, csv_file, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.data_dir = data_dir
        self.dataset = CarImageDataset(data_dir=self.data_dir, csv_file=csv_file, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

# debugging
if __name__ == '__main__':
    from utils.util import get_parent_path
    parent_path = get_parent_path()
    data_dir = get_parent_path() / "data"
    car_loader = CarImageDataLoader(data_dir=data_dir, csv_file='train_mapped.csv', batch_size=32, training=True)
    
    for images, labels in car_loader:
        print(images.shape, labels.shape)
        break  # Just to test the first batch