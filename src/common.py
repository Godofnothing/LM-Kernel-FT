import torch.nn.functional as F

__all__ = ["target_transform"]

def target_transform(targets, num_classes: int):
    return F.one_hot(targets, num_classes=num_classes).float()
