import torch
import os

class CFG:
    """
    Configuration class for all hyperparameters and settings.
    """
    project_name = "Ensemble-ConvNext-XGBoost"
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data Paths
    data_dir = "/workspace/img_clf/data/"
    train_csv_path = data_dir + "train_csv/train_mapped_3.csv"
    test_csv_path = data_dir + "test.csv"
    train_image_dir = data_dir + "train"
    test_image_dir = data_dir + "test"
    output_dir = './outputs/'

    # Data preprocessing and augmentation
    img_size = (224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Model Configuration
    # For ConvNext feature extractor
    model_name = 'convnext_tiny.fb_in22k'
    pretrained = True
    num_classes = 396

    # Training Hyperparameters
    epochs = 30
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-6
    patience = 5  # For early stopping
    smoothing = 0.1

    # Cross-validation
    n_splits = 5

    # k-folds
    n_fold = 5

    # XGBoost Parameters
    xgb_params = {
        'objective': 'multi:softmax', # or 'multi:softprob'
        'num_class': num_classes,
        'eval_metric': 'mlogloss',
        'eta': 0.1,
        'max_depth': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': seed
    }
print("Configuration loaded:")
print(f"  Device: {CFG.device}")
print(f"  Model: {CFG.model_name}")
print(f"  Image Size: {CFG.img_size}")