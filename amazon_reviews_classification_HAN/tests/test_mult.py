import torch
import pytest

from torch import nn


def matrix_mul(seq, weight, bias):
    s_list = []
    for s in seq:
        s = torch.mm(s, weight)
        if bias is not None:
            s = s + bias.expand(s.size()[0], bias.size()[1])
        s = s.unsqueeze(0)
        s_list.append(s)
    return torch.cat(s_list, 0).squeeze()


def element_wise_mul(inp1, inp2):
    feat_list = []
    for feat1, feat2 in zip(inp1, inp2):
        feat2 = feat2.unsqueeze(1).expand_as(feat1)
        feat = feat1 * feat2
        feat_list.append(feat.unsqueeze(0))
    output = torch.cat(feat_list, 0)
    return torch.sum(output, 0).unsqueeze(0)


torch.manual_seed(0)


@pytest.mark.parametrize(
    "seq, weight, bias",
    [
        (torch.rand(32, 10, 100), torch.rand(100, 100), None),
        (torch.rand(32, 10, 100), torch.rand(100, 100), torch.rand(100)),
    ],
)
def test_matmul(seq, weight, bias):
    if bias is not None:
        ll = nn.Linear(weight.shape[0], weight.shape[1])
        ll.bias.data = bias
        bias = bias.unsqueeze(0)
    else:
        ll = nn.Linear(weight.shape[0], weight.shape[1], bias=False)
    ll.weight.data = weight
    out1 = ll(seq)
    out2 = matrix_mul(seq, torch.transpose(weight, 1, 0), bias)
    assert out1.allclose(out2)


torch.manual_seed(1)


def test_element_wise():
    inp1 = torch.rand(10, 32, 100)
    inp2 = torch.rand(10, 32)
    out1 = (inp2.unsqueeze(2) * inp1).sum(0).unsqueeze(0)
    out2 = element_wise_mul(inp1, inp2)
    assert out1.allclose(out2)
