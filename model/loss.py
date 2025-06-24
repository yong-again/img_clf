import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np


def nll_loss(output, target):
    return F.nll_loss(output, target)

def CrossEntropy(num_classes: int = None, label_index_list: list = None, device: str = 'cpu', use_weight: bool = False):
    """
    Returns a CrossEntropyLoss criterion, optionally using class weights to handle class imbalance.

    Args:
        num_classes (int, optional): Total number of classes in the classification task.
                                     Required if use_weight is True.
        label_index_list (list, optional): List of label indices for all training samples.
                                           Used to compute class weights based on frequency.
        device (str, optional): Device to place the computed class weights on. Default is 'cpu'.
        use_weight (bool, optional): If True, computes and applies class weights using sklearn's
                                     'balanced' strategy to mitigate class imbalance.

    Returns:
        nn.CrossEntropyLoss: A PyTorch cross-entropy loss function, weighted if specified.

    Raises:
        ValueError: If `use_weight` is True but `num_classes` or `label_index_list` is not provided.
    """
    if use_weight:
        if num_classes is None or label_index_list is None:
            raise ValueError("num_classes and label_index_list must be provided when use_weight is True.")
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
    """
    Cross-entropy loss with label smoothing using PyTorch's built-in support.

    Label smoothing helps prevent the model from becoming overconfident by
    distributing some probability mass to all other classes, not just the correct class.

    Args:
        smoothing (float): The smoothing factor. A value between 0 and 1,
                           where 0 means no smoothing (standard cross-entropy).
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, input, target):
        """
        Compute the smoothed cross-entropy loss.

        Args:
            input (torch.Tensor): Logits from the model. Shape: (batch_size, num_classes).
            target (torch.Tensor): Ground truth labels. Shape: (batch_size,).

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.criterion(input, target)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing more on hard-to-classify examples.

    This loss reduces the relative loss for well-classified examples, putting more focus
    on hard, misclassified examples.

    Args:
        alpha (float): Weighting factor for the rare class. Default is 1.
        gamma (float): Focusing parameter that reduces loss for well-classified examples.
        reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss between `inputs` and the ground truth `targets`.

        Args:
            inputs (torch.Tensor): Logits from the model. Shape: (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels. Shape: (batch_size,).

        Returns:
            torch.Tensor: The computed focal loss.
        """
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
