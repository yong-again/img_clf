import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import sys
sys.path.append('..')
from base import BaseModel, ResNet
import warnings 
warnings.filterwarnings("ignore")


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ResNetclassifier(ResNet):
    def forward(self, x):
        return self.backbone(x)


class SwinClassfier(nn.Module):
    def __init__(self, num_classes=396):
        super().__init__()
        self.backbone = create_model('swin_base_patch4_window7_224', 
                                     pretrained=True, 
                                     num_classes=num_classes, 
                                     global_pool='avg')
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
class ConvNextClassifier(nn.Module):
    def __init__(self, num_classes=396):
        super().__init__()
        self.backbone = create_model('convnext_base', 
                                     pretrained=True, 
                                     num_classes=num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
class VitClassifier(nn.Module):
    def __init__(self, num_classes=396):
        super().__init__()
        self.backbone = create_model('vit_base_patch16_224', 
                                     pretrained=True, 
                                     num_classes=num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

class CustomConvNextClassifier(nn.Module):
    def __init__(self, num_classes=396, **kwargs):
        super().__init__()

        # 1. init backbone
        self.backbone = create_model(
            'convnext_base',
            pretrained=True,
            num_classes=0  # Feature extractor 모드
        )

        # 2.custom head
        in_features = 1024  # convnext_base의 출력 차원
        self.head = self._build_head_layers(
            kwargs.get('head_layers', []),
            in_features
        )

    def _build_head_layers(self, layer_configs, in_features):
        layers = []
        current_dim = in_features

        for config in layer_configs:
            layer_type = config["type"]
            args = config.get("args", {})

            if layer_type == "LayerNorm":
                layers.append(nn.LayerNorm(**args))
            elif layer_type == "Linear":
                args["in_features"] = current_dim
                out_features = args["out_features"]
                layers.append(nn.Linear(**args))
                current_dim = out_features
            elif layer_type == "GELU":
                layers.append(nn.GELU())
            elif layer_type == "Dropout":
                layers.append(nn.Dropout(**args))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
