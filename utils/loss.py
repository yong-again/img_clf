import torch
from torch import nn
from torch.nn import functional as F

class LossSelector(nn.Module):
    def __init__(self, loss_type='cross_entropy'):
        super(LossSelector, self).__init__()
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'bce':
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)