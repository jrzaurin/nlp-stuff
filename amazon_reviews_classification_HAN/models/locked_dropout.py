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
