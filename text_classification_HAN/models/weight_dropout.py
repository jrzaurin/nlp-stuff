"""
Code here is taken DIRECTLY from the awd-lstm-lm Salesforce repo:
https://github.com/salesforce/awd-lstm-lm/tree/32fcb42562aeb5c7e6c9dec3f2a3baaaf68a5cb5

Credit to the authors: Stephen Merity, Nitish Shirish Keskar and Richard Socher
"""

import torch
import warnings

from torch import nn


class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False, verbose=True):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self.verbose = verbose
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            if self.verbose:
                print("Applying weight drop of {} to {}".format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + "_raw", nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + "_raw")
            w = None
            if self.variational:
                mask = torch.ones(raw_w.size(0), 1)
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=self.training)
                w = nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = nn.Parameter(
                    torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
                )
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)
