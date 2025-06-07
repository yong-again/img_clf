import torch.nn as nn
import numpy as np
from abc import abstractmethod
from torchvision import models

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class ResNet(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, num_classes=396):
        super(ResNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        
        in_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feature, num_classes)
        
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)