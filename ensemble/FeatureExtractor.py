import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CFG
from EnsembleModel import ConvNextFeatureExtractor
from data_loader import ImageDataset
from utils import seed_everything

def extract_features(model, data_loader, device):
    """
    Extracts features and labels from a data_loader using the given model.

    Args:
        model (nn.Module): The feature extractor model.
        data_loader (DataLoader): The DataLoader for the dataset.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: A tuple containing (features, labels).
               Features is a numpy array of shape (num_samples, num_features).
               Labels is a numpy array of shape (num_samples,).
    """
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for image, labels in tqdm(data_loader, desc="Extracting features"):
            images = image.to(device)
            # Get the feature vectors from the model
            features = model(images)

            # Move features and labels to CPU and convert to Numpy
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches into single numpy arrays
    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    return features_array, labels_array


def main():
    """
    main function to run the feature extraction pipeline
    """
    seed_everything(CFG.seed)

    # --- 1. Load Data ---
    # We will extract features from the entire training dataset at once.
    # The Csv should have 'img_path' and 'label' columns.

    try:
        df = pd.read_csv(CFG.train_csv_path)
    except FileNotFoundError:
        print(f"Error : training CSV not found at {CFG.train_csv_path}")
        print("Please create a dummy CSV for testing with columns 'img_path' and 'label'.")
        # Create dummy dataframe for demonstration purpose
        dummy_data = {
            'img_path' : [f'dummy_path_{i}.png' for i in range(100)],
            'label' : np.random.randint(0, CFG.num_classes, 100)
        }
        df = pd.DataFrame(dummy_data)
        print("Using a dummy DataFrame")

    # --- 2. Initialize Model ---
    print("Initializing the feature extractor model ....")
    model = ConvNextFeatureExtractor(model_name=CFG.model_name, pretrained=CFG.pretrained)
    model.to(CFG.device)

    # --- 3. Create DataLoader ---
    # We use a 'test' type transform because we don't want to apply
    # augmentations during feature extraction.
    dataset = ImageDataset(df, data_type='test')
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    # --- 4. Extract Features ---
    print("Starting feature extraction ...")
    features, labels = extract_features(model, data_loader, CFG.device)\

    print(f"\nExtracted features shape: {features.shape}")
    print(f"Extracted features labels: {labels.shape}")

    # --- 5. save Features ---
    # Ensure the output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Save the featurs and labels to compressed Numpy file
    output_path = os.path.join(CFG.output_dir, f"{CFG.model_name}_features.npy")
    np.savez_compressed(output_path, features=features, labels=labels)

    print(f"\nFeatures and labels succesfully saved to {output_path}")

if __name__ == "__main__":
    main()


