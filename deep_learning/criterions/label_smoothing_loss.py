"""
Label Smoothing formula:
y_ls = (1 - alpha) * y_hot + alpha / k
where y_hot is the probability represented in one-hot encoding

With a bit of math, LS_Loss can be calculated as:
(1 - alpha) * CE_Loss + alpha * mean(log_probabilities)

If we add another constant term log(1/num_classes) to LS_Loss, it can then be written as
(1 - alpha) * CE_Loss + alpha * KL_Div(u, p)
where u is a uniform distribution on the number of classes.

Article for reference: https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
"""

import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha):
        if alpha < 0 or alpha > 1.:
            raise ValueError('alpha should be a value between 0 and 1')
        super(LabelSmoothingLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        if self.training:
            log_prob = F.log_softmax(output, dim=-1)
            nll_loss = -log_prob * target
            nll_loss = nll_loss.sum(-1)
            l1_loss = nll_loss.squeeze(1).sum()
            kl_div_loss = -log_prob.mean(-1)
            return l1_loss * (1. - self.alpha) + self.alpha * kl_div_loss
        else:
            return F.cross_entropy(output, target)
