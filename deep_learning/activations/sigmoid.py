"""Implement the standard and hard sigmoid functions"""

import torch.nn as nn
import torch.nn.functional as F


class Sigmoid(nn.Module):
    def __init__(self, hard=False):
        super(Sigmoid, self).__init__()
        self.hard = hard

    def forward(self, x):
        if self.hard:
            return F.relu6(
                x + 3.
            ) * 0.16667
        else:
            return F.sigmoid(x)
