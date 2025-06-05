# 🚗 HAI! - Hecto AI Challenge 2025: 중고차 이미지 차종 분류

본 프로젝트는 헥토(Hecto) 주관의 AI 경진대회에 참가하여 **중고차 이미지 기반 차종 분류 모델**을 개발한 결과물입니다.  
PyTorch 기반으로 EfficientNet 및 ResNet 아키텍처를 활용하였고, 데이터 불균형 및 클래스 유사성 문제를 고려하여 모델을 설계했습니다.

---

## 📝 대회 개요

- **주최/주관**: Hecto / Dacon
- **목표**: 실제 중고차 이미지를 기반으로 총 396개 차량 차종 분류
- **평가 지표**: Log Loss (Cross Entropy)
- **제약 사항**
  - 외부 데이터 사용 불가
  - 일부 유사 클래스는 동일 클래스로 간주 (사전 정제 필요)
  - 2025년 5월 19일 이전 공개된 pretrained 모델만 사용 가능

---

## 📂 프로젝트 구조

``````
img_clf/
│
├── train.py - training script for model training
├── test.py - evaluation of trained model
│
├── config.json - holds configuration for training
├── parse_config.py - class to handle config file and cli options
│
├── new_project.py - initialize new project with template files
│
├── base/ - abstract base classes
│   ├── base_data_loader.py
│   ├── base_model.py
│   └── base_trainer.py
│
├── data_loader/ - anything about data loading goes here
│   └── data_loaders.py
│
├── data/ - default directory for storing input data
│
├── model/ - models, losses, and metrics
│   ├── model.py
│   ├── metric.py
│   └── loss.py
│
├── saved/
│   ├── models/ - trained models are saved here
│   └── log/ - default logdir for tensorboard and logging output
│
├── trainer/ - trainers
│   └── trainer.py
│
├── logger/ - module for tensorboard visualization and logging
│   ├── visualization.py
│   ├── logger.py
│   └── logger_config.json
│  
└── utils/ - small utility functions
    ├── util.py
    └── ...
``````