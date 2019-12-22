import numpy as np
import pdb
import torch
import torch.nn.functional as F

from torch import nn

use_cuda = torch.cuda.is_available()


def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
      padding_idx = -1

  X = torch.nn.functional.embedding(words, masked_embed_weight,
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  return X


class HierAttnNet(nn.Module):
    def __init__(self,
        vocab_size,
        maxlen_sent,
        maxlen_doc,
        word_hidden_dim=32,
        sent_hidden_dim=32,
        padding_idx=1,
        embed_dim=50,
        embed_dropout=0.,
        embedding_matrix=None,
        num_class=4,
        init_method='zeros'):
        super(HierAttnNet, self).__init__()

        self.word_hidden_dim = word_hidden_dim
        self.init_method = init_method
        self.wordattnnet = WordAttnNet(
            vocab_size=vocab_size,
            hidden_dim=word_hidden_dim,
            padding_idx=padding_idx,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            embedding_matrix=embedding_matrix
            )

        self.sentattnnet = SentAttnNet(
            word_hidden_dim=word_hidden_dim,
            sent_hidden_dim=sent_hidden_dim,
            padding_idx=padding_idx
            )

        self.fc = nn.Linear(sent_hidden_dim*2, num_class)

    def _init_hidden(self, init_method, batch_size):
        if init_method is 'zeros':
            return nn.init.zeros_(torch.Tensor(2, batch_size, self.word_hidden_dim))
        elif init_method is 'kaiming_normal':
            return nn.init.kaiming_normal_(torch.Tensor(2, batch_size, self.word_hidden_dim))

    def forward(self, X):
        x = X.permute(1,0,2)
        word_h_n = self._init_hidden(self.init_method, X.shape[0])
        if use_cuda: word_h_n = word_h_n.cuda()
        sent_list = []
        for sent in x:
            out, word_h_n = self.wordattnnet(sent, word_h_n)
            sent_list.append(out)
        doc = torch.cat(sent_list, 1)
        out = self.sentattnnet(doc)
        return self.fc(out)


class RNNAttn(nn.Module):
    def __init__(self,
        vocab_size,
        maxlen,
        num_layers=3,
        hidden_dim=32,
        rnn_dropout=0.,
        padding_idx=1,
        embed_dim=50,
        embed_dropout=0.,
        embedding_matrix=None,
        num_class=4,
        with_attention=False):
        super(RNNAttn, self).__init__()

        self.embed_dropout = embed_dropout
        self.with_attention = with_attention

        if isinstance(embedding_matrix, np.ndarray):
            self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1], padding_idx = padding_idx)
            self.word_embed.weight = nn.Parameter(torch.Tensor(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx = padding_idx)

        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=True,
            batch_first=True
            )

        self.fc = nn.Linear(hidden_dim*2, num_class)

        if self.with_attention:
            self.attn = Attention(hidden_dim*2, maxlen)

    def forward(self, X):
        if self.embed_dropout > 0.:
            embed = embedded_dropout(self.word_embed, X.long(), dropout=self.embed_dropout if self.training else 0)
        else:
            embed = self.word_embed(X.long())
        o, (h, c) = self.rnn(embed)
        if self.with_attention:
            out = self.attn(o)
        else:
            out = torch.cat((h[-2], h[-1]), dim = 1)
        return self.fc(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim, seq_len):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.bias   = nn.Parameter(torch.zeros(seq_len))

    def forward(self, inp):
        x = inp.contiguous().view(-1, self.hidden_dim)
        u = torch.tanh_(torch.mm(x, self.weight).view(-1, self.seq_len) + self.bias)
        a = F.softmax(u, dim=1)
        self.attn_weights = a
        s = (inp * torch.unsqueeze(a, 2)).sum(1)
        return s


class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        self.attn  = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, inp):
        u = torch.tanh_(self.attn(inp))
        a = F.softmax(self.contx(u), dim=1)
        self.attn_weights = a
        s = (a * inp).sum(1)
        return s


class WordAttnNet(nn.Module):
    def __init__(self,
        vocab_size,
        hidden_dim=32,
        padding_idx=1,
        embed_dim=50,
        embed_dropout=0.,
        embedding_matrix=None):
        super(WordAttnNet, self).__init__()

        self.embed_dropout = embed_dropout

        if isinstance(embedding_matrix, np.ndarray):
            self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1], padding_idx = padding_idx)
            self.word_embed.weight = nn.Parameter(torch.Tensor(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.word_attn = AttentionWithContext(hidden_dim*2)

    def forward(self, X, h_n):
        if self.embed_dropout > 0.:
            x = embedded_dropout(self.word_embed, X.long(), dropout=self.embed_dropout if self.training else 0)
        else:
            x = self.word_embed(X.long())
        h_t, h_n = self.rnn(x, h_n)
        s = self.word_attn(h_t).unsqueeze(1)
        return s, h_n


class SentAttnNet(nn.Module):
    def __init__(self,
        word_hidden_dim=32,
        sent_hidden_dim=32,
        padding_idx=1):
        super(SentAttnNet, self).__init__()

        self.rnn = nn.GRU(word_hidden_dim*2, sent_hidden_dim, bidirectional=True,
            batch_first=True)
        self.sent_attn = AttentionWithContext(sent_hidden_dim*2)

    def forward(self, X):
        h_t, h_n = self.rnn(X)
        v = self.sent_attn(h_t)
        return v