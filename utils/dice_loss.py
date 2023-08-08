
import torch


def dice_loss(outputs, targets, smooth=1e-6):
    intersection = torch.sum(outputs * targets)
    union = torch.sum(outputs) + torch.sum(targets)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_score