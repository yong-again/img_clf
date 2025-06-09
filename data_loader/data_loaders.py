from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
import sys
sys.path.append('../')
from base import BaseDataLoader
from utils.util import get_parent_path
from data_loader.transform import train_transform, valid_transform, train_transform_albu, valid_transform_albu, AlbumentationsWithAugMix
import numpy as np

import logging

# UTF-8 인코딩으로 로그 파일 핸들러 지정
log_file_path = 'error_image_paths.log'
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

logger = logging.getLogger('image_error_logger')
logger.setLevel(logging.ERROR)
logger.addHandler(file_handler)
logger.propagate = False  # 루트 로거로 전달 방지

class CarImageDataset(Dataset):
    """
    Car Image Classification dataset
    """
    def __init__(self, data_dir, csv_file, return_onehot=False, transform=None):
        self.data_dir = data_dir
        self.csv_file = os.path.join(get_parent_path(), data_dir, 'train_csv', csv_file)
        self.return_onehot = return_onehot
        self.transform = transform
        self.data_frame = pd.read_csv(self.csv_file)
        self.num_classes = len(self.data_frame['label_index'].unique())
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        try:
            row = self.data_frame.iloc[idx]
            img_path = os.path.basename(row['image_path'])
            folder_name = row['folder_name']
            img_full_path = f"{get_parent_path()}/{self.data_dir}/train/{folder_name}/{img_path}"
            image = Image.open(img_full_path).convert('RGB')
            image = np.array(image)
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
                
            label = row['label_index']
            if self.return_onehot:
                label = torch.eye(self.num_classes)[label]
            else:    
                label = torch.tensor(label, dtype=torch.long)
            
            return image, label
        
        except Exception as e:
            logging.error(f"{img_full_path} | idx={idx} | error={str(e)}")
            return None
        
    def clone_with_transform(self, transform):
        return CarImageDataset(
            data_dir=self.data_dir,
            csv_file=self.csv_file,
            transform=transform,
        )

class CarImageDataLoader(BaseDataLoader):
    """
    Car Image Classfication dataset
    """
    def __init__(self, data_dir, csv_file, batch_size, shuffle=True, validation_split=0.0, num_workers=2, return_onehot=False):
        trsfm = train_transform_albu()
        self.data_dir = data_dir
        self.dataset = CarImageDataset(data_dir=self.data_dir, csv_file=csv_file, return_onehot=return_onehot, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
class CarAugImageDataset(Dataset):
    """
    Car Image Classification dataset
    """
    def __init__(self, data_dir, csv_file, return_onehot=False, transform=None):
        self.data_dir = data_dir
        self.csv_file = os.path.join(get_parent_path(), data_dir, 'train_csv', csv_file)
        self.return_onehot = return_onehot
        self.transform = transform
        self.data_frame = pd.read_csv(self.csv_file)
        self.num_classes = len(self.data_frame['label_index'].unique())
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        try:
            row = self.data_frame.iloc[idx]
            img_path = os.path.basename(row['image_path'])
            folder_name = row['folder_name']
            img_full_path = f"{get_parent_path()}/{self.data_dir}/train/{folder_name}/{img_path}"
            image = Image.open(img_full_path).convert('RGB')
            image = np.array(image)
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
                
            
            label = int(row['label_index'])
            
            if self.return_onehot:
                label = torch.eye(self.num_classes)[label]
                print(f"One-hot encoded label shape: {label.shape}")
            else:    
                label = torch.tensor(label, dtype=torch.long)
                # print(f"Label shape: {label.shape}")
                # print(f"Label value: {label.item()}")
                
                
            if label is None:
                raise ValueError(f"Label is None for idx={idx} in {img_full_path}")

            return image, label
        
        except Exception as e:
            logging.error(f"{img_full_path} | idx={idx} | error={str(e)}")
            return None
        
    def clone_with_transform(self, transform):
        return CarAugImageDataset(
            data_dir=self.data_dir,
            csv_file=self.csv_file,
            transform=transform,
        )

class CarAugImageDataLoader(BaseDataLoader):
    """
    Car Image Classfication dataset
    """
    def __init__(self, data_dir, csv_file, batch_size, shuffle=True, validation_split=0.0, num_workers=2, return_onehot=False):
        trsfm = train_transform_albu()
        self.data_dir = data_dir
        self.dataset = CarAugImageDataset(data_dir=self.data_dir, csv_file=csv_file, return_onehot=return_onehot, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CarTestDataSet(Dataset):
    """
    Car Image Classification dataset for testing
    """
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.csv_file = os.path.join(get_parent_path(), data_dir, csv_file)
        self.transform = transform
        self.data_frame = pd.read_csv(self.csv_file)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        file_name = os.path.basename(row['img_path'])
        img_full_path = os.path.join(get_parent_path(), self.data_dir, 'test', file_name)
        
        image = Image.open(img_full_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, file_name

class CarTestDataLoader(BaseDataLoader):
    """
    Car Image Classification dataset for testing
    """
    def __init__(self, data_dir, csv_file, batch_size, shuffle=False, validation_split=0.0, num_workers=2):
        trsfm = valid_transform_albu()
        self.data_dir = data_dir
        self.dataset = CarTestDataSet(data_dir=self.data_dir, csv_file=csv_file, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

# debugging
if __name__ == '__main__':
    from utils.util import get_parent_path
    parent_path = get_parent_path()
    data_dir = get_parent_path() / "data" 
    car_loader = CarAugImageDataLoader(data_dir=data_dir, csv_file='train_mapped_2.csv', batch_size=32)
    
    for images, labels in car_loader:
        print(images.shape, labels.shape)
        break