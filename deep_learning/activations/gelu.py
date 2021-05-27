'''A custom implementation of the Gaussian Error Linear Unit'''

import math
import torch
import torch.nn as nn

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()
    
    def forward(self, x):
        cdf = 0.5 * (1 + torch.tanh(
            (math.sqrt(2 / math.pi) * (x + 0.044715) * torch.pow(x, 3))
        ))

        return x * cdf
