'''
Implement the Google Swish activation function
The Swish activation is simply x * sigmoid(x).
Similar to the sigmoid function, Swish comes with
a soft and a hard version as well.
'''

import torch.nn as nn
import torch.functional as F

class Swish(nn.Module):
    def __init__(self, hard=False):
        super(Swish, self).__init__()
        self.hard = hard
    
    def forward(self, x):
        if self.hard:
            return x * F.relu6(
                x + 3.
            ) * 0.16667
        else:
            return x * F.sigmoid(x)
