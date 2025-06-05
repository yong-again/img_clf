import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')  # Adjust the path as necessary
from base import BaseModel, ResNet


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
    
# debugging
if __name__ == '__main__':
    import torch
    model = ResNetclassifier(num_classes=391)
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)  # Example forward pass