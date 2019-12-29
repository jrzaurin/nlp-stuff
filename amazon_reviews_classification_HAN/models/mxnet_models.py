import numpy as np
import mxnet as mx

from mxnet import nd, init
from mxnet.gluon import nn, rnn, Block
from gluonnlp.model.utils import apply_weight_drop


import pdb

ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()


class HierAttnNet(Block):
    def __init__(
        self,
        vocab_size,
        maxlen_sent,
        maxlen_doc,
        word_hidden_dim=32,
        sent_hidden_dim=32,
        padding_idx=1,
        embed_dim=50,
        weight_drop=0.0,
        embed_drop=0.0,
        locked_drop=0.0,
        last_drop=0.0,
        embedding_matrix=None,
        num_class=4,
    ):
        super(HierAttnNet, self).__init__()

        self.word_hidden_dim = word_hidden_dim

        with self.name_scope():
            self.wordattnnet = WordAttnNet(
                vocab_size=vocab_size,
                hidden_dim=word_hidden_dim,
                embed_dim=embed_dim,
                weight_drop=weight_drop,
                embed_drop=embed_drop,
                locked_drop=locked_drop,
                embedding_matrix=embedding_matrix,
            )

            self.sentattnnet = SentAttnNet(
                word_hidden_dim=word_hidden_dim,
                sent_hidden_dim=sent_hidden_dim,
                weight_drop=weight_drop,
            )

            self.ld = nn.Dropout(last_drop)
            self.fc = nn.Dense(in_units=sent_hidden_dim * 2, units=num_class)

    def forward(self, X):
        x = X.transpose(axes=(1, 0, 2))
        word_h_n = nd.zeros(shape=(2, X.shape[0], self.word_hidden_dim), ctx=ctx)
        word_a_list, word_s_list = [], []
        for sent in x:
            word_a, word_s, word_h_n = self.wordattnnet(sent, word_h_n)
            word_a_list.append(word_a)
            word_s_list.append(word_s)
        self.sent_a = nd.concat(*word_a_list, dim=1)
        sent_s = nd.concat(*word_s_list, dim=1)
        doc_a, doc_s = self.sentattnnet(sent_s)
        self.doc_a = doc_a.transpose(axes=(0, 2, 1))
        doc_s = self.ld(doc_s)
        return self.fc(doc_s)


class RNNAttn(Block):
    def __init__(
        self,
        vocab_size,
        maxlen,
        num_layers=3,
        hidden_dim=32,
        rnn_dropout=0.0,
        padding_idx=1,
        embed_dim=50,
        embed_drop=0.0,
        locked_drop=0.0,
        last_drop=0.0,
        embedding_matrix=None,
        num_class=4,
        with_attention=False,
    ):
        super(RNNAttn, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.with_attention = with_attention

        self.embed_drop = embed_drop
        self.locked_drop = locked_drop

        with self.name_scope():
            self.word_embed, embed_dim = get_embedding(
                vocab_size, embed_dim, embed_drop, locked_drop, embedding_matrix
            )

            self.rnn = rnn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=rnn_dropout,
                bidirectional=True,
                layout="NTC",
            )

            if self.with_attention:
                self.attn = Attention(hidden_dim * 2, maxlen)

            self.ld = nn.Dropout(last_drop)
            self.fc = nn.Dense(in_units=hidden_dim * 2, units=num_class)

    def forward(self, X):
        embed = self.word_embed(X)
        h0 = nd.zeros(shape=(self.num_layers * 2, X.shape[0], self.hidden_dim), ctx=ctx)
        c0 = nd.zeros(shape=(self.num_layers * 2, X.shape[0], self.hidden_dim), ctx=ctx)
        o, (h, c) = self.rnn(embed, [h0, c0])
        if self.with_attention:
            self.attn_w, out = self.attn(o)
        else:
            out = nd.concat(*[h[-2], h[-1]], dim=1)

        out = self.ld(out)
        return self.fc(out)


class WordAttnNet(Block):
    def __init__(
        self,
        vocab_size,
        hidden_dim=32,
        embed_dim=50,
        weight_drop=0.0,
        embed_drop=0.0,
        locked_drop=0.0,
        embedding_matrix=None,
    ):
        super(WordAttnNet, self).__init__()

        self.embed_drop = embed_drop
        self.locked_drop = locked_drop
        self.weight_drop = weight_drop

        with self.name_scope():
            self.word_embed, embed_dim = get_embedding(
                vocab_size, embed_dim, embed_drop, locked_drop, embedding_matrix
            )

            self.rnn = rnn.GRU(
                input_size=embed_dim, hidden_size=hidden_dim, bidirectional=True, layout="NTC",
            )
            if weight_drop:
                apply_weight_drop(self.rnn, ".*h2h_weight", rate=weight_drop)

            self.word_attn = AttentionWithContext(hidden_dim * 2)

    def forward(self, X, h_n):
        x = self.word_embed(X)
        h_t, h_n = self.rnn(x, h_n)
        a, s = self.word_attn(h_t)
        return a, s.expand_dims(1), h_n


class SentAttnNet(Block):
    def __init__(self, word_hidden_dim=32, sent_hidden_dim=32, padding_idx=1, weight_drop=0.0):
        super(SentAttnNet, self).__init__()

        with self.name_scope():
            self.rnn = rnn.GRU(
                input_size=word_hidden_dim * 2,
                hidden_size=sent_hidden_dim,
                bidirectional=True,
                layout="NTC",
            )
            if weight_drop:
                apply_weight_drop(self.rnn, ".*h2h_weight", rate=weight_drop)

            self.sent_attn = AttentionWithContext(sent_hidden_dim * 2)

    def forward(self, X):
        h_t = self.rnn(X)
        a, v = self.sent_attn(h_t)
        return a, v


class Attention(Block):
    def __init__(self, hidden_dim, seq_len):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        with self.name_scope():
            self.weight = nd.random.normal(shape=(hidden_dim, 1), ctx=ctx)
            self.bias = nd.zeros(seq_len, ctx=ctx)

    def forward(self, inp):
        x = inp.reshape(-1, self.hidden_dim)
        u = nd.tanh(nd.dot(x, self.weight).reshape(-1, self.seq_len) + self.bias)
        a = nd.softmax(u, axis=1)
        s = (inp * a.expand_dims(2)).sum(1)
        return a, s


class AttentionWithContext(Block):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        with self.name_scope():
            self.attn = nn.Dense(in_units=hidden_dim, units=hidden_dim, flatten=False)
            self.contx = nn.Dense(in_units=hidden_dim, units=1, flatten=False, use_bias=False)

    def forward(self, inp):
        u = nd.tanh(self.attn(inp))
        a = nd.softmax(self.contx(u), axis=1)
        s = (a * inp).sum(1)
        return a.transpose(axes=(0, 2, 1)), s

def get_embedding(vocab_size, embed_dim, embed_drop, locked_drop, embedding_matrix):
    embedding = nn.HybridSequential()
    with embedding.name_scope():
        if isinstance(embedding_matrix, np.ndarray):
            embedding_block = nn.Embedding(vocab_size, embedding_matrix.shape[1])
            embedding_block.weight.set_data(nd.from_numpy(embedding_matrix))
            embed_dim = embedding_matrix.shape[1]
        else:
            embedding_block = nn.Embedding(
                vocab_size, embed_dim, weight_initializer=init.Uniform(0.1)
            )
        if embed_drop:
            apply_weight_drop(embedding_block, "weight", embed_drop, axes=(1,))
        embedding.add(embedding_block)
        if locked_drop:
            embedding.add(nn.Dropout(locked_drop, axes=(0,)))
    return embedding, embed_dim
