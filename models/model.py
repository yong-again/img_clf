# load model architecture
import torch
from torch import nn, optim
from torchvision import transforms

class ModelWeight(nn.Module):
    def __init__(self, model_name, num_classes=88):
        super(ModelWeight, self).__init__()
        self.model = get_model(model_name, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def get_model(model_name, num_classes=88):
        if model_name == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'resnet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        return model