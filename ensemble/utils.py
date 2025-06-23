import albumentations as A
from Warraper import AlbumentationsWithAugMix
from sklearn.model_selection import StratifiedKFold, GroupKFold
from config import CFG
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
import torch
import random
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def get_transforms(*, data_type: str):
    """
    Returns the appropriate set of image transformations based on the data type.
    Args:
        data_type (str): Either 'train' or 'valid'.
    Returns:
        A.Compose or AlbumentationsWithAugMix: The augmentation pipeline.
    """
    if data_type == 'train':
        # Base augmentations before AugMix and normalization
        base_transform = A.Compose([
            A.Resize(CFG.img_size[0] + 32, CFG.img_size[1] + 32), # Start with a slightly larger image
            A.RandomResizedCrop(size=(CFG.img_size[0],CFG.img_size[1]), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.1, alpha_coef=0.1),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2)),
                A.RandomRain(p=1.0),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=7),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MedianBlur(blur_limit=5),
            ], p=0.3),
            A.ImageCompression(quality_range=(30, 70), p=0.2),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 8),
                            hole_height_range=(1, int(CFG.img_size[0] / 8)),
                            hole_width_range=(0, int(CFG.img_size[1] / 8)),
                            p=0.5),
        ])

        # Normalization and tensor conversion
        normalize_transform = A.Compose([
            A.Normalize(mean=CFG.mean, std=CFG.std),
            ToTensorV2()
        ])

        # The user-specified wrapper for AugMix
        return AlbumentationsWithAugMix(base_transform, normalize_transform)

    elif data_type == 'valid' or data_type == 'test':
        # Validation/Test transforms: only resize, normalize, and convert to tensor
        return A.Compose([
            A.Resize(CFG.img_size[0], CFG.img_size[1]),
            A.Normalize(mean=CFG.mean, std=CFG.std),
            ToTensorV2(),
        ])

def split_data(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    """
    Splits the dataframe into K-folds using GroupKFold to ensure that
    all images from a single subject are in the same fold.

    Args:
        df (pd.DataFrame): The dataframe containing file paths and labels.
        group_col (str): The column name to group by (e.g., 'subject_id').
        target_col (str): The column name for the target label. It is required by
                          the scikit-learn splitter's API but not used for stratification
                          in GroupKFold.

    Returns:
        pd.DataFrame: The input dataframe with a new 'fold' column.
    """
    folder = GroupKFold(n_splits=CFG.n_splits)

    # Initialize 'fold' column if it doesn't exist
    df['fold'] = -1

    groups = df[group_col]
    targets = df[target_col]

    for fold_num, (train_idx, val_idx) in enumerate(folder.split(df, targets, groups)):
        df.loc[val_idx, 'fold'] = fold_num

    df['fold'] = df['fold'].astype(int)
    print("Data split into folds using GroupKFold:")
    print(df['fold'].value_counts())
    return df

def data_split(folds):
    Fold = GroupKFold(n_splits=CFG.n_fold)
    group = folds['subject_id']
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds["Atelectasis"], group)):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    return folds


def seed_everything(seed=None):
    """
    Sets the random seed for reproducibility across all relevant libraries.
    Args:
        seed (int): The seed value. If None, uses the seed from CFG.
    """
    if seed is None:
        seed = CFG.seed

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # For reproducibility, you might also want to set benchmark to False
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_parent_path():
    """
    Get the path of the parent path script.
    """
    return Path(__file__).parent.parent.resolve()


