import torch
from torch._C import CONV_BN_FUSION
import torch.nn as nn
import torch.nn.functional as F

'''
GoogLeNet was invented by Google used for image
classification.

It is well-known for its usage of Inception, a block
of convolutional layers which enables the network to be 
much deeper than vanilla CNN, hence achieving higher performance.
'''

__all__ = ['GoogleNet']

class BasicConv2d(nn.Module):
    '''a basic convolutional block'''

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=0.001)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        return F.relu(out, inplace=True)


class Inception(nn.Module):
    '''Build the Inception Block'''

    def __init__(
        self,
        in_channels,
        ch1_1,
        ch3_3red,
        ch3_3,
        ch5_5red,
        ch5_5,
        pool_proj,
        conv_block=None
    ):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1_1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3_3red, kernel_size=1),
            conv_block(ch3_3red, ch3_3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5_5red, kernel_size=1),
            conv_block(ch5_5red, ch5_5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
    
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (4,4))
        # size: in_channels * 4 * 4
        out = self.conv(out)
        # size: 128 * 4 * 4
        out = torch.flatten(out, 1)
        # size: 2048
        out = self.fc1(out)
        out = F.relu(out, inplace=True)
        out = F.dropout(out, 0.7, training=self.training)

        out = self.fc2(out)
        return out

    

class GoogLeNet(nn.Module):
    def __init__(
        self,
        num_classes,
        aux_logits=False,
        transform_input=False
    ):
        super(GoogLeNet, self).__init__()
        blocks = [BasicConv2d, Inception, InceptionAux]