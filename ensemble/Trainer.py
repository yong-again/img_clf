import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class Trainer:
    """
    
    """
    def __init__(self, model, classifier, criterion, optimizer, device, writer: SummaryWriter):
        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = writer

    def train_one_epoch(self, data_loader, epoch: int):
        self.model.train()
        self.classifier.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(data_loader, desc=f"Training Epoch {epoch}")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            features = self.model(images)
            outputs = self.classifier(features)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        if self.writer is not None:
            self.writer.add_scalar("loss/train", avg_loss, epoch)
            self.writer.add_scalar("accuracy/train", accuracy, epoch)
            self.writer.add_scalar("f1/train", f1, epoch)

        return avg_loss, accuracy, f1

    def validate_one_epoch(self, data_loader, epoch: int):
        self.model.eval()
        self.classifier.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Validating Epoch {epoch}")

            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # --- Forward pass ---
                features = self.model(images)
                outputs = self.classifier(features)

                loss = self.criterion(outputs, labels)

                # --- Logging and Metrics ---
                total_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        if self.writer is not None:
            self.writer.add_scalar("loss/valid", avg_loss, epoch)
            self.writer.add_scalar("accuracy/valid", accuracy, epoch)
            self.writer.add_scalar("f1/valid", f1, epoch)

        return avg_loss, accuracy, f1
