import torch
import torch.nn as nn

def adapt_target_for_loss(output, target, loss_fn):
    """
    Adapt the target for the loss function if necessary.
    This function checks if the loss function requires the target to be in a specific format
    - CrossEntropyLoss expects target as class indices
    - other : one-hot encoded targets or float format

    Args:
        output (Tensor): Model output predictions.
        target (Tensor): Ground truth labels.(usually in one-hot format)
        loss_fn (nn.Module): module representing the loss function.
    """
    
    # CrossEntropyLoss or NLLLoss expects target as class indices
    if isinstance(loss_fn, (nn.CrossEntropyLoss, nn.NLLLoss)):
        if target.ndim == 2: # one-hot encoded
            return target.argmax(dim=1)
        else:
            return target # already in class indices format
    else:
        return target # for other loss functions, return target as is