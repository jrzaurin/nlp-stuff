import numpy as np
import mxnet as mx
import pdb

from mxnet import gluon, autograd, nd, init
from mxnet.gluon import nn, rnn, HybridBlock

from gluonnlp.model.utils import apply_weight_drop


class HierAttnNet(HybridBlock):
    def __init__(self,
        vocab_size,
        maxlen_sent,
        maxlen_doc,
        word_hidden_dim=32,
        sent_hidden_dim=32,
        rnn_dropout=0.,
        padding_idx=1,
        embed_dim=50,
        embed_dropout=0.,
        embedding_matrix=None,
        num_class=4,
        init_method='zeros'):
        super(HierAttnNet, self).__init__()

        self.word_hidden_dim = word_hidden_dim

        with self.name_scope():
            self.wordattnnet = WordAttnNet(
                vocab_size=vocab_size,
                hidden_dim=word_hidden_dim,
                rnn_dropout=rnn_dropout,
                padding_idx=padding_idx,
                embed_dim=embed_dim,
                embed_dropout=embed_dropout,
                embedding_matrix=embedding_matrix
                )

            self.sentattnnet = SentAttnNet(
                word_hidden_dim=word_hidden_dim,
                sent_hidden_dim=sent_hidden_dim,
                rnn_dropout=rnn_dropout,
                padding_idx=padding_idx
                )

            self.fc = nn.Dense(in_units=sent_hidden_dim*2, units=num_class)

    def forward(self, X):
        x = X.transpose(axes=(1,0,2))
        word_h_n = nd.zeros(shape=(2, X.shape[0], self.word_hidden_dim))
        sent_list = []
        for sent in x:
            out, word_h_n = self.wordattnnet(sent, word_h_n)
            sent_list.append(out)
        doc = nd.concat(*sent_list, dim=1)
        out = self.sentattnnet(doc)
        return self.fc(out)


class RNNAttn(HybridBlock):
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

        self.with_attention = with_attention

        with self.name_scope():
            if isinstance(embedding_matrix, np.ndarray):
                self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1])
                self.word_embed.weight.set_data(nd.from_numpy(embedding_matrix))
                embed_dim = embedding_matrix.shape[1]
            else:
                self.word_embed = nn.Embedding(vocab_size, embed_dim,
                    weight_initializer=init.Uniform(0.1))
            if embed_dropout:
                apply_weight_drop(self.word_embed, 'weight', embed_dropout, axes=(1,))

            self.rnn = rnn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                bidirectional=True,
                layout='NTC'
                )

            self.fc = nn.Dense(in_units=hidden_dim*2, units=num_class)

            if self.with_attention:
                self.attn = Attention(hidden_dim*2, maxlen)

    def forward(self, X):
        self.word_embed.weight._data[self.padding_idx] = 0.
        embed = self.word_embed(X)
        o, (h, c) = self.rnn(embed)
        if self.with_attention:
            out = self.attn(o)
        else:
            out = nd.concatenate([h[-2], h[-1]], axis=1)
        return self.fc(out)


class Attention(HybridBlock):
    def __init__(self, hidden_dim, seq_len):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        with self.name_scope():
            self.weight = nd.random.normal(shape=(hidden_dim, 1))
            self.bias   = nd.zeros(seq_len)

    def forward(self, inp):
        x = inp.reshape(-1, self.hidden_dim)
        u = nd.tanh(nd.dot(x, self.weight).reshape(-1, self.seq_len) + self.bias)
        a = nd.softmax(u, dim=1)
        self.attn_weights = a
        s = (inp * a.expand_dims(2)).sum(1)
        return s


class AttentionWithContext(HybridBlock):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        with self.name_scope():
            self.attn  = nn.Dense(in_units=hidden_dim, units=hidden_dim, flatten=False)
            self.contx = nn.Dense(in_units=hidden_dim, units=1, flatten=False, use_bias=False)

    def forward(self, inp):
        u = nd.tanh(self.attn(inp))
        a = nd.softmax(self.contx(u), axis=1)
        self.attn_weights = a
        s = (a * inp).sum(1)
        return s


class WordAttnNet(HybridBlock):
    def __init__(self,
        vocab_size,
        hidden_dim=32,
        rnn_dropout=0.,
        padding_idx=1,
        embed_dim=50,
        embed_dropout=0.,
        embedding_matrix=None):
        super(WordAttnNet, self).__init__()

        self.padding_idx = padding_idx

        with self.name_scope():
            if isinstance(embedding_matrix, np.ndarray):
                self.word_embed = nn.Embedding(vocab_size, embedding_matrix.shape[1])
                self.word_embed.weight.set_data(nd.from_numpy(embedding_matrix))
                embed_dim = embedding_matrix.shape[1]
            else:
                self.word_embed = nn.Embedding(vocab_size, embed_dim,
                    weight_initializer=init.Uniform(0.1))

            self.rnn = rnn.GRU(input_size=embed_dim, hidden_size=hidden_dim, bidirectional=True,
                layout='NTC')
            self.word_attn = AttentionWithContext(hidden_dim*2)

    def forward(self, X, h_n):
        x = self.word_embed(X)
        h_t, h_n = self.rnn(x, h_n)
        s = self.word_attn(h_t).expand_dims(1)
        return s, h_n


class SentAttnNet(HybridBlock):
    def __init__(self,
        word_hidden_dim=32,
        sent_hidden_dim=32,
        rnn_dropout=0.,
        padding_idx=1):
        super(SentAttnNet, self).__init__()

        with self.name_scope():
            self.rnn = rnn.GRU(input_size=word_hidden_dim*2, hidden_size=sent_hidden_dim,
                bidirectional=True, layout='NTC')
            self.sent_attn = AttentionWithContext(sent_hidden_dim*2)

    def forward(self, X):
        h_t = self.rnn(X)
        v = self.sent_attn(h_t)
        return v