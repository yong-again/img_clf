import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
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