"""
Code here is taken DIRECTLY from the awd-lstm-lm Salesforce repo:
https://github.com/salesforce/awd-lstm-lm/tree/32fcb42562aeb5c7e6c9dec3f2a3baaaf68a5cb5

Credit to the authors: Stephen Merity, Nitish Shirish Keskar and Richard Socher
"""

from torch import nn


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        mask = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
