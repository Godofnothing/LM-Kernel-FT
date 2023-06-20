import torch

@torch.no_grad()
def accuracy(preds, targets):
    if len(targets.shape) > 1:
        targets = targets.argmax(dim=-1)
    return (preds.argmax(dim=-1) == targets).float().mean()
