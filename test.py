import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
import argparse
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn


class TestDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df['img_path'].tolist()
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_path = os.path.join('data', img_path)  # 경로 조정
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, os.path.basename(img_path)  # 이미지와 파일명 반환

def main(args):
    # 변환 정의 (훈련 시 사용한 것과 동일해야 함)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 및 로더 생성
    test_dataset = TestDataset(args.csv_path, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 모델 로드 (훈련 시 사용한 모델 클래스와 동일해야 함)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1000)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    
    # 추론 실행
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for images, batch_filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            filenames.extend(batch_filenames)

    # 결과 저장
    result_df = pd.DataFrame({
        'image_path': filenames,
        'prediction': predictions
    })
    result_df.to_csv(args.output_path, index=False)
    print(f"결과가 {args.output_path}에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='훈련된 모델 경로')
    parser.add_argument('--csv_path', type=str, required=True, help='테스트 CSV 파일 경로')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='결과 저장 경로')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    args = parser.parse_args()
    
    main(args)