"""
Implements the Log Softmax Activation.

Implementation details follow that of PyTorch.
"""

import torch
import torch.nn as nn

class LogSoftmax(nn.Module):
    def __init__(self):
        super(LogSoftmax, self).__init__()
    
    def forward(self, x):
        '''Use a different formula to calculate log(softmax)'''
        return x - torch.max(x)
