import torch
import torch.nn as nn
from torch.optim import Adam



class BaseTrainer:
    def __init__(self, model, criterion, optimizer=None, device='cpu'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer if optimizer else Adam(self.model.parameters())
        self.device = device
        
        