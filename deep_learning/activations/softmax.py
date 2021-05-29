'''Implements the softmax layer'''

import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
    
    def forward(self, x):
        exp = torch.exp(x)
        return exp / torch.sum(exp)
