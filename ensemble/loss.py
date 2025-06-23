import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implements label-smoothed cross-entropy loss.
    Label smoothing is a regularization technique that prevents the model from
    becoming too confident in its predictions, which can help generalization.

    Instead of using a one-hot vector for the true label (e.g., [0, 1, 0]),
    it uses a smoothed vector (e.g., [0.05, 0.9, 0.05]).
    """
    def __init__(self, classes, smoothing=0.1, dim=-1):
        """
        Args:
            classes (int): The number of classes.
            smoothing (float): The smoothing factor (epsilon).
            dim (int): The dimension over which to apply softmax.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): The model's raw output (logits).
                                    Shape: (batch_size, num_classes).
            target (torch.Tensor): The ground truth labels.
                                    Shape: (batch_size,).
        """
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # create the smoothed target distribution
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.confidence / self.cls)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # calculate the KL divergence loss
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FocalLoss(nn.Module):
    """
    Implements Focal Loss, designed to address class imbalance.
    It down-weights the loss assigned to well-classified examples, focusing
    training on a sparse set of hard examples.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Args:
            alpha (float): The weighting factor for the classes. Can be a list.
            gamma (float): The focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): The model's raw output (logits).
                                    Shape: (batch_size, num_classes).
            targets (torch.Tensor): The ground truth labels.
                                    Shape: (batch_size,).
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

def get_loss_fn(loss_name: str, **kwargs):
    """
    Factory function to get a loss function by name.
    """
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    elif loss_name == 'focal_loss':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

if __name__ == '__main__':
    # --- Test Label Smoothing ---
    print("Testing Label Smoothing Loss...")
    ls_loss_fn = get_loss_fn('label_smoothing', classes=5, smoothing=0.1)
    dummy_pred = torch.randn(4, 5)  # 4 samples, 5 classes
    dummy_target = torch.randint(0, 5, (4,))
    ls_loss = ls_loss_fn(dummy_pred, dummy_target)
    print(f"Label Smoothing Loss: {ls_loss.item()}\n")

    # --- Test Focal Loss ---
    print("Testing Focal Loss...")
    focal_loss_fn = get_loss_fn('focal_loss', alpha=1, gamma=2)
    fl_loss = focal_loss_fn(dummy_pred, dummy_target)
    print(f"Focal Loss: {fl_loss.item()}\n")

    # --- Test Cross Entropy ---
    print("Testing Cross Entropy Loss...")
    ce_loss_fn = get_loss_fn('cross_entropy')
    ce_loss = ce_loss_fn(dummy_pred, dummy_target)
    print(f"Cross Entropy Loss: {ce_loss.item()}\n")
