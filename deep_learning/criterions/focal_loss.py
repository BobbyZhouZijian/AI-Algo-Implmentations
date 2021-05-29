'''Implements Focal Loss'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    
    def forward(self, x, targets, alpha=0.8, gamma=2):
        x = F.sigmoid(x)
        alpha = torch.tensor([alpha, 1-alpha])

        # flatten
        x = x.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(x, targets, reduction='mean')
        BCE_exp = torch.exp(-BCE)

        focal_loss = alpha * (1 - BCE_exp)**gamma * BCE
        return focal_loss