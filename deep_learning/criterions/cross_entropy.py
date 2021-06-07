"""
Implements the Cross Entropy Loss

Cross Entropy loss is calculated as:

Loss(x, y) = \sum px * log(px)
where px = x * y

We calculate log(px) using log softmax
and then apply nll_loss to log(px) to get
the cross entropy loss.
"""

import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, x, target):
        return F.nll_loss(
            F.log_softmax(x, 1),
            target,
            self.weight,
            None,
            self.ignore_index,
            None,
            self.reduction
        )
