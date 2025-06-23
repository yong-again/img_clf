import os
import gc
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Import all our custom modules
from config import CFG
from utils import seed_everything, split_data
from data_loader import ImageDataset, create_dataloader
from EnsembleModel import ConvNextFeatureExtractor
from loss import get_loss_fn
from Trainer import Trainer

def run_finetuning():
    """
    Stage 1: Fine-tunes the ConvNext feature extractor using K-fold cross validation.
    For each fold, it trains a model and saves the version that performs best on the
    validation set. It also extracts and saves the out-of-fold (OOF) features.
    """
    print("-- Stage 1: Starting Feature Extractor Fine-Tuning")
    seed_everything(CFG.seed)

    # --- 1. Load and prepare data ---
    try:
        df = pd.read_csv(CFG.train_csv_path)
    except FileNotFoundError:
        print(f"Error: Training CSV not Found at {CFG.train_csv_path}. PLease check it")
        return None

    df_folds = split_data(df, group_col='label_index', target_col='label')

    oof_features = []
    oof_labels = []

    # --- 2. K-Fold Cross-Validation Loop ---
    for fold in range(CFG.n_fold):
        print(f"\n========== Fold: {fold} ==========")

        # --- 2.1 setup for the current fold ---
        log_dir = os.path.join(CFG.output_dir, f'log/fold_{fold}')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # Create Dataloaders
        train_loader, valid_loader = create_dataloader(df_folds, fold=fold)

        # Initialize Model, Classifier and optimizer
        model = ConvNextFeatureExtractor(model_name=CFG.model_name, pretrained=True).to(CFG.device)
        classifier = nn.Linear(model.n_features, CFG.num_classes).to(CFG.device)

        # combine parameters from both model for the optimizer
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(classifier.parameters()),
            lr=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
        )

        criterion = get_loss_fn('label_smoothing', classes=CFG.num_classes, smoothing=CFG.smoothing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)

        # Initialize Trainer
        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=CFG.device, writer=writer)

        # --- 2.2 Epoch Loop ---
        best_val_f1 = 0.0
        patience_counter = 0
        best_epoch = -1

        for epoch in range(CFG.epochs):
            train_loss, train_acc, train_f1 = trainer.train_one_epoch(train_loader, epoch)
            val_loss, val_acc, val_f1 = trainer.validate_one_epoch(valid_loader, epoch)

            print(f"Epoch {epoch}: Train Loss: {train_loss: .4f}, Train F1: {train_f1: .4f} | Valid Loss: {val_loss: .4f}, Valid F1: {val_f1: .4f}")

            scheduler.step()

            # --- 2.3 Model Checkpointing and Early Stopping ---
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                # save the best model weights for this fold
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'classifier_state_': classifier.state_dict(),
                }, os.path.join(CFG.output_dir, f'best_model_fold_{fold}.pth'))
                print(f"Best F1 score improved to {best_val_f1: .4f}. Saving model")

            else:
                patience_counter += 1

            if patience_counter >= CFG.patience:
                print(f"Early Stopping at epoch: {epoch} as F1 did not improved for {CFG.patience} epochs.")
                break

        writer.close()

        # --- 2.4. Extract OOF features with the best model for this fold ---
        print(f"Extracting OOF features for fold {fold} using from epoch {best_epoch}")
        best_model_path = os.path.join(CFG.output_dir, f'best_model_fold_{fold}.pth')
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        fold_features = []
        fold_labels = []

        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Extracting OOF features for fold {fold}"):
                images = images.to(CFG.device)
                features = model(images)
                fold_features.append(features.cpu().numpy())
                fold_labels.append(labels.cpu().numpy())

        oof_features = np.concatenate(fold_features)
        oof_labels = np.concatenate(fold_labels)

        # Clean up Memory
        del model, classifier, optimizer, scheduler, trainer, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

        # ---3. Save all OOF features ---
        print("\nAll folds proceed. Saving combined OOF features.")
        all_oof_features = np.concatenate(oof_features)
        all_oof_labels = np.concatenate(oof_labels)

        oof_save_path = os.path.join(CFG.output_dir, 'oof_features.npz')
        np.savez_compressed(oof_save_path, features=all_oof_features, labels=all_oof_labels)
        print(f"OOF Feature saved to {oof_save_path}")

def train_xgboost():
    """
    Stage 2 : Trains an XGBoost model out-of-fold (OOF) features
    generated during the fine-tuning stage.
    """
    print("\n --- Stage 2: Starting XGBoost Training ---")

    # ---1. Load OOF Features ---
    oof_path = os.path.join(CFG.output_dir, 'oof_features.npz')
    try:
        data = np.load(oof_path)
        X_train = data['features']
        y_train = data['labels']
    except FileNotFoundError:
        print(f"Error: OOF Features not found at {oof_path}. Please run fine-tuning first")
        return

    print(f"Loaded OOF Features shape: {X_train.shape}, Labels shape: {y_train.shape}")

    # --- 2. Train XGBoost ---
    print("Training XGBoost model...")

    xgb_classifier = xgb.XGBClassifier(**CFG.xgb_params)

    xgb_classifier.fit(X_train, y_train)

    # --- 3.Evaluate and Save ---
    # Evaluate on the training data itself ( since these are OOF, it's a valid measure)
    preds = xgb_classifier.predict(X_train)
    accuracy = accuracy_score(y_train, preds)
    f1 = f1_score(y_train, preds, average='macro')

    print("\n--- XGBoost Model Performance on OOF data ---")
    print(f" Accuracy: {accuracy: .4f}")
    print(f" Macro F1-score: {f1: .4f}")

    # Save the trained XGBoost model
    xgb_model_path = os.path.join(CFG.output_dir, 'xgboost_model.joblib')
    joblib.dump(xgb_classifier, xgb_model_path)
    print(f"\nSaved trained XGBoost model to: {xgb_model_path}")

if __name__ == '__main__':
    run_finetuning()
    train_xgboost()




