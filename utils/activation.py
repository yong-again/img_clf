import torch 
import torch.nn as nn
import torch.nn.functional as F


class LoadActivation(nn.Module):
    def __init__(self, activation_type='relu'):
        super(Activation, self).__init__()
        self.activation_type = activation_type.lower()
        
        if self.activation_type == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif self.activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif self.activation_type == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def forward(self, x):
        return self.activation(x)