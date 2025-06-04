import os
import sys
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from utils.loader import ImageDataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

class TransformedSubset(Subset):
    """변환을 적용하는 서브셋 클래스"""
    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        if self.transform:
            x = self.transform(x)
        return x, y

def load_data(csv_dir):
    dataset = ImageDataset(csv_dir=csv_dir, transform=None)
    labels = [label for _, label in dataset]

    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=labels,  # 클래스 분포 유지
        random_state=42
    )
    assert max(train_indices) < len(dataset), f"훈련 인덱스가 데이터셋 범위를 벗어났습니다: {max(train_indices)} >= {len(dataset)}"
    assert max(val_indices) < len(dataset), f"검증 인덱스가 데이터셋 범위를 벗어났습니다: {max(val_indices)} >= {len(dataset)}"

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TransformedSubset(Subset(dataset, train_indices), train_transform)
    val_dataset = TransformedSubset(Subset(dataset, val_indices), val_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )

    train_counts = np.bincount([labels[i] for i in train_indices])
    val_counts = np.bincount([labels[i] for i in val_indices])
    print(f"훈련 세트 클래스 당 최소 샘플: {min(train_counts)}")
    print(f"검증 세트 클래스 당 최소 샘플: {min(val_counts)}")

    return train_dataloader, val_dataloader

def log_images(writer, images, epoch, max_images=16):
    """TensorBoard에 이미지 그리드 로깅"""
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    images = images[:max_images]
    images = torch.stack([inv_normalize(img) for img in images])
    grid = make_grid(images, nrow=4)
    writer.add_image('train/images', grid, epoch)

def run(model, train_loader, val_loader=None, epochs=10, learning_rate=0.001, device=None, save_dir='exp2'):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # TensorBoard 설정 (exp2 디렉토리)
    writer = SummaryWriter(log_dir=save_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_progress = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx, (images, labels) in enumerate(train_progress):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if epoch == 0 and batch_idx == 0:
                log_images(writer, images, epoch)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            train_progress.set_postfix({
                'loss': running_loss/(train_progress.n+1),
                'acc': 100.*correct/total
            })

        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        if val_loader:
            model.eval()
            val_progress = tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{epochs}', leave=False)
            with torch.no_grad():
                for images, labels in val_progress:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
                    val_progress.set_postfix({
                        'val_loss': val_loss/(val_progress.n+1),
                        'val_acc': 100.*val_correct/val_total
                    })

            val_loss = val_loss/len(val_loader)
            val_acc = 100.*val_correct/val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{save_dir}/best_model.pth')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        if val_loader:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        scheduler.step()

        log = f"Epoch [{epoch+1}/{epochs}] | "
        log += f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
        if val_loader:
            log += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        print(log)

    writer.close()
    torch.save(model.state_dict(), f'{save_dir}/final_model.pth')
    return history

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    csv_dir = r'C:\works\dacon\img_clf\data\train_csv\train_file_list.csv'
    train_loader, val_loader = load_data(csv_dir)

    # 모델 초기화 (클래스 수 자동 반영)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(train_loader.dataset.classes))

    history = run(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        learning_rate=1e-3,
        save_dir='exp2'
    )
    print("Training complete.")
