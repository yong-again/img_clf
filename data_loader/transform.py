from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import AugMix, CutMix
import warnings
warnings.filterwarnings("ignore")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def train_transform():
    """
    Returns the transformation for training dataset.
    """
    return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
def valid_transform():
    """
    Returns the transformation for validation dataset.
    """
    return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                ])

def train_transform_albu():
    return A.Compose([
        A.Resize(256, 256),  # 원본 비율 보존 및 일관된 입력 크기
        A.RandomResizedCrop((224, 224), scale=(0.8, 1.0), p=1.0),

        # 기본 변형
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),

        # 컬러 관련 변형
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),

        # 날씨 및 환경 효과
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2),
            A.RandomRain(p=1.0),
        ], p=0.2),

        # 노이즈 계열
        A.OneOf([
            A.MotionBlur(blur_limit=7),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),

        # JPEG 압축 노이즈
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.2),

        # 조도 및 명암 대비 향상
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),

        # Random Erasing 유사 효과
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),

        # Normalize 및 tensor 변환
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
def valid_transform_albu():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
class AlbumentationsWithAugMix:
    def __init__(
        self,
        base_transform: A.Compose,
        normalize_transform: A.Compose,
        severity: int = 3,
        mixture_width: int = 3,
        alpha: float = 1.0
    ):
        # base_transform: Resize, RandomResizedCrop, Flip, etc (np.ndarray → np.ndarray)
        # normalize_transform: Normalize + ToTensorV2 (np.ndarray → Tensor)
        self.base_transform = base_transform
        self.normalize_transform = normalize_transform
        self.severity = severity
        self.mixture_width = mixture_width
        self.alpha = alpha

    def __call__(self, image: np.ndarray):
        orig = image.copy()
        orig_t = self.base_transform(image=orig)['image'].astype(np.float32)

        # Dirichlet (벡터) → numpy array
        ws = np.random.dirichlet([self.alpha] * self.mixture_width).astype(np.float32)
        # Beta (스칼라) → float or np.float32
        m  = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = np.zeros_like(orig_t, dtype=np.float32)
        for i in range(self.mixture_width):
            aug = image.copy()
            aug_t = self.base_transform(image=aug)['image'].astype(np.float32)
            mix += ws[i] * aug_t

        mixed = (1 - m) * orig_t + m * mix
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return self.normalize_transform(image=mixed)
        
def train_transform_augmix():
    base_transform = A.Compose([
        A.Resize(256, 256),  # 원본 비율 보존 및 일관된 입력 크기
        A.RandomResizedCrop((224, 224), scale=(0.8, 1.0), p=1.0),

        # 기본 변형
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),

        # 컬러 관련 변형
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),

        # 날씨 및 환경 효과
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2),
            A.RandomRain(p=1.0),
        ], p=0.2),

        # 노이즈 계열
        A.OneOf([
            A.MotionBlur(blur_limit=7),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),

        # JPEG 압축 노이즈
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.2),

        # 조도 및 명암 대비 향상
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),

        # Random Erasing 유사 효과
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        
        ])
    
    normalize_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
        
    return AlbumentationsWithAugMix(base_transform, normalize_transform)
            
            
# if __name__ == '__main__':
#     # Example usage
#     img = Image.open(r'/workspace/img_clf/data/train/2시리즈_액티브_투어러_F45_2019_2021/2시리즈_액티브_투어러_F45_2019_2021_0000.jpg')
#     transform = AlbumentationWithCutMix()
#     transformed_img = transform(img)
#     print(transformed_img.shape)  # Should print the shape of the transformed tensor