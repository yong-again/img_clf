import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
import argparse
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from pathlib import Path

class TestDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df['img_path'].tolist()  # 컬럼명 확인
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_path = os.path.join('data', img_path)
        image = Image.open(img_path).convert('RGB')  # 전체 경로 사용
        
        if self.transform:
            image = self.transform(image)
            
        return image, os.path.basename(img_path)

def main(args):
    # 1. 변환 정의 (훈련과 동일)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 데이터셋 및 로더 생성
    test_dataset = TestDataset(args.csv_path, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 3. 모델 초기화 (전이 학습 구조 동일하게)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3-1. 사전 학습된 모델 구조 로드
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # 3-2. 클래스 매핑 정보 로드
    class_to_idx = torch.load('class_mapping.pth')  # 훈련 시 저장된 파일
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 3-3. FC 레이어 교체 (훈련 시와 동일한 구조)
    model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
    
    # 3-4. 체크포인트 로드
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # 4. 추론 실행
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for images, batch_filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 클래스 인덱스를 실제 라벨로 변환
            batch_labels = [idx_to_class[idx.item()] for idx in preds]
            predictions.extend(batch_labels)
            filenames.extend(batch_filenames)

    # 5. 결과 저장
    result_df = pd.DataFrame({
        'image_path': filenames,
        'prediction': predictions
    })
    result_df.to_csv(args.output_path, index=False)
    print(f"결과가 {args.output_path}에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, 
                       help='훈련된 모델 경로 (예: exp2/best_model.pth)')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='테스트 CSV 파일 경로 (image_path 컬럼 포함)')
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                       help='결과 저장 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기 (기본값: 32)')
    args = parser.parse_args()
    
    # 경로 검증
    assert Path(args.model_path).exists(), f"모델 파일 없음: {args.model_path}"
    assert Path(args.csv_path).exists(), f"CSV 파일 없음: {args.csv_path}"
    
    main(args)
