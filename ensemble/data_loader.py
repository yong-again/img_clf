import os
import torch
import cv2
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from config import CFG
from utils import get_transforms

class ImageDataset(Dataset):
    """
    Custom Pytorch Dataset for loading image and labels from a dataframe.
    """
    def __init__(self, df: pd.DataFrame, data_type: str ='train'):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and labels.
            data_type (str): The type of data, e.g., 'train', 'valid'.
                                This determines which augmentations to apply.
        """
        self.df = df
        self.image_path = df['image_path'].values if data_type == 'test' else df['img_path'].values
        self.folder_name = df['folder_name'].values
        self.image_name = df['image_name'].values
        self.labels = df['label_index'].values if 'label' in df.columns else None
        self.data_type = data_type
        self.transform = get_transforms(data_type=self.data_type)

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the specified index.
        """
        # Construct the full image path

        img_path = os.path.join(CFG.data_dir, 'train', self.folder_name[idx], self.image_name[idx])
        # Read the image using OpenCV and convert from BGR to RGB
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not find image at {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder tensor if an image is corrupt
            return torch.randn(3, CFG.img_size[0], CFG.img_size[1]), torch.tensor(-1)

        # Apply Transform
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Get the label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            # For test set, where there are no labels
            return image

def create_dataloader(df: pd.DataFrame, fold: int):
    """
    Creates training and validation dataloaders for a given fold.

    Args:
        df (pd.DataFrame): The full dataframe with a 'fold' column.
        fold (int): The current fold number to use for validation.

    Returns:
        tuple: A tuple containing (train_loader, valid_loader).
    """
    # split dataframe into training and validation sets
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)

    print(f"Fold {fold}:")
    print(f"  Training samples {len(train_df)} ")
    print(f"  Validation samples {len(valid_df)} ")

    # Create datasets
    train_dataset = ImageDataset(train_df, data_type='train')
    valid_dataset = ImageDataset(valid_df, data_type='valid')

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False
    )
    return train_loader, valid_loader

if __name__ == '__main__':
    import pandas as pd

    train_df = pd.read_csv(CFG.train_csv_path)
    train_dataset = ImageDataset(train_df, data_type='train')

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True
    )

    for image, label in train_loader:
        print(f"image shape {image.shape}")
        print(f"label shape {label.shape}")
        break