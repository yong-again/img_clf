import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np


def nll_loss(output, target):
    return F.nll_loss(output, target)

def CrossEntropy(num_classes=None, label_index_list=None, device='cpu', use_weight=False):
    if use_weight:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),
            y=label_index_list
        )
        weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    else:
        return nn.CrossEntropyLoss()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, input, target):
        return self.criterion(input, target)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss