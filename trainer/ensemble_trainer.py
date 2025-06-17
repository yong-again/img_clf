# run_train_pipeline.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import FeatureExtractor, extract_features
from logger import log_xgboost_metrics
import importlib

def get_dataset_from_config(config: dict, split: str = "train", val_transform=None):
    """
    config를 기반으로 특정 split(train/val)에 해당하는 Dataset or DataLoader를 생성
    Args:
        config (dict): config["data_loader"] 딕셔너리
        split (str): "train" 또는 "val"
        val_transform: validation 시 적용할 transform 함수 (예: valid_transform_albu())
    Returns:
        torch.utils.data.Dataset 또는 DataLoader
    """
    data_loader_type = config["type"]
    data_loader_args = config.get("args", {})

    # DataLoader 클래스 동적 로딩
    module = importlib.import_module("data_loader.data_loaders")
    cls = getattr(module, data_loader_type)
    loader_instance = cls(**data_loader_args)

    if split == "train":
        return loader_instance
    elif split == "val":
        return loader_instance.split_validation(val_transform=val_transform)
    else:
        raise ValueError(f"Invalid split: {split}")

def run_train_pipeline(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = config.get_logger("train")
    writer = SummaryWriter(log_dir=config["trainer"]["log_dir"])

    # 1. Load datasets and dataloaders
    train_dataset = get_dataset_from_config(config["data_loader"], split="train")
    test_dataset = get_dataset_from_config(config["data_loader"], split="val")
    train_loader = DataLoader(train_dataset, batch_size=config["data_loader"]["args"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["data_loader"]["args"]["batch_size"], shuffle=False)

    # 2. Initialize feature extractor
    feature_model = FeatureExtractor(config["feature_model"]).to(device)
    feature_model.eval()

    logger.info("Extracting train features...")
    X_train, y_train = extract_features(feature_model, train_loader, device, writer, tag_prefix="train")

    logger.info("Extracting test features...")
    X_test, y_test = extract_features(feature_model, test_loader, device, writer, tag_prefix="val")

    # 3. Train ensemble model
    logger.info("Training ensemble model...")
    ensemble_model = get_ensemble_model(config["ensemble_model"], **config.get("xgb_params", {}))
    ensemble_model.fit(X_train, y_train)

    # 4. Save ensemble model
    save_path = os.path.join(config["trainer"]["save_dir"], config["name"])
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, "xgb_model.joblib")
    joblib.dump(ensemble_model, model_save_path)
    logger.info(f"XGBoost model saved to {model_save_path}")

    # 5. Predict and log
    logger.info("Evaluating on test set...")
    y_pred = ensemble_model.predict(X_test)
    y_prob = ensemble_model.predict_proba(X_test)

    log_xgboost_metrics(writer, y_test, y_pred, y_prob)

    writer.close()
    logger.info(f"Experiment complete. Results and logs saved in {save_path}")

    return ensemble_model, y_pred, y_test
