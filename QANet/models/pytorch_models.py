import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

d_model = config.d_model
n_head = config.num_heads
d_word = config.glove_dim
d_char = config.char_dim
batch_size = config.batch_size
dropout = config.dropout
dropout_char = config.dropout_char

d_k = d_model // n_head
d_cq = d_model * 4
len_c = config.para_limit
len_q = config.ques_limit


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)


class PositionalEncodingTransformer(nn.Module):
    """Implement the PE function."""
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionalEncodingQANet(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncodingQANet, self).__init__()

        freqs = torch.Tensor(
            [10000. ** (-i / d_model) if i % 2 == 0 else -10000. ** ((1 - i) / d_model) for i in range(d_model)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(max_len).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()

        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=False)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=False)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception(" Parameter 'dim' must have value 1 or 2")

        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        if bias:
            nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):
    def __init__(self, num_layers, size, non_linearity, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.T = nn.ModuleList([nn.Linear(size, size) for _ in range(self.num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.num_layers)])
        self.non_linearity = non_linearity

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in range(self.num_layers):
            T = torch.sigmoid(self.T[layer](x))
            H = self.non_linearity(self.linear[layer](x))
            H = F.dropout(H, p=dropout, training=self.training)
            x = H * T + x * (1 - T)
        x = x.transpose(1, 2)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        bsize = x.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(bsize, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(bsize, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Embedding(nn.Module):
    def __init__(self, wlinear=False, depthwise=True):
        super().__init__()

        self.wlinear = wlinear
        self.depthwise = depthwise

        if depthwise:
            self.conv2d = DepthwiseSeparableConv(d_char, d_model, 5, dim=2)
        else:
            self.conv2d = nn.Conv2d(d_char, d_model, kernel_size = 5, padding=0, bias=True)
            nn.init.kaiming_normal_(self.conv2d.weight)

        if self.wlinear:
            self.linear = nn.Linear(d_word+d_char, d_model)
            self.high = Highway(2, d_model)
        else:
            self.high = Highway(2, d_word+d_char)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb) if self.depthwise else F.relu(self.conv2d(ch_emb))
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()

        wd_emb = F.dropout(wd_emb, p=dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)

        emb = torch.cat([ch_emb, wd_emb], dim=1)
        if self.wlinear:
            emb = self.high(self.linear(emb))
        return emb


class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, num_conv_layers, num_channels, k, max_len):
        super().__init__()

        self.pos_encoder = PositionalEncodingTransformer(max_len)
        self.sublayers = clones(SublayerConnection(d_model, dropout), total_sublayers)
        self.convs = nn.ModuleList([DepthwiseSeparableConv(num_channels, num_channels, k) for _ in range(num_conv_layers)])
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.FF = FeedForward(num_channels, num_channels)
        self.norm_1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv_layers)])
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

    def forward(self, x, mask, l, num_blocks):
        total_sublayers = (num_conv_layers + 2) * num_blocks
        out = PosEncoder(x)

        # sublayer 1: LayerNorm -> conv -> residual connection
        l = 1
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_1[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self._layer_dropout(out, res, dropout*float(l)/total_sublayers)
            l += 1

        # sublayer 2: LayerNorm -> Attention -> residual connection
        res = out
        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_attn(out, mask)
        out = self._layer_dropout(out, res, dropout*float(l)/total_layers)
        l += 1

        # sublayer 3: LayerNorm -> FeedForward -> residual connection
        res = out
        out = self.norm_3(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FF(out)
        out = self._layer_dropout(out, res, dropout*float(l)/total_layers)

        return out

    def _layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()

        w = torch.empty(d_model * 3)
        w_c = torch.empty(d_model, 1)
        w_q = torch.empty(d_model, 1)
        w_mul = torch.empty(1, 1, d_model)
        bias = torch.empty(1)
        self.w_c = nn.Parameter(nn.init.xavier_uniform_(w))
        self.w_c = nn.Parameter(nn.init.xavier_uniform_(w_c))
        self.w_c = nn.Parameter(nn.init.xavier_uniform_(w_q))
        self.w_c = nn.Parameter(nn.init.xavier_uniform_(w_mul))
        self.bias = nn.Parameter(nn.init.constant_(bias, 0))


    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        Cmask = Cmask.unsqueeze(2)
        Qmask = Qmask.unsqueeze(1)
        batch_size_c = C.size()[0]
        if self.optimized:
            S = self._optimized_trilinear_for_attention(C, Q)
        else:
            S = self._trilinear_for_attention(C, Q)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def _trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        CQ = torch.mul(C, Q)
        subres = torch.cat([C, Q, CQ], dim=3)
        res = torch.matmul(subres, self.w)
        return res

    def _optimized_trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w_c).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w_q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w_mul, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()

        w1 = torch.empty(d_model * 2)
        w2 = torch.empty(d_model * 2)
        self.w1 = nn.Parameter(nn.init.xavier_uniform_(w1))
        self.w2 = nn.Parameter(nn.init.xavier_uniform_(w2))

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = torch.matmul(self.w1, X1)
        Y2 = torch.matmul(self.w2, X2)
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2


class QANet(nn.Module):
    def __init__(self, word_mat=None, char_mat=None, word_vocab=None, char_vocab=None):
        super().__init__()

        if word_mat is not None:
            self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat),
                freeze=config.freeze)
        else:
            self.word_emb = nn.Embedding(len(word_vocab.itos), config.word_emb_size)

        if char_mat is not None:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat),
                freeze=False)
        else:
            self.char_emb = nn.Embedding(len(char_vocab.itos), config.char_emb_size)

        self.emb = Embedding()
        self.c_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_c)
        self.q_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_q)
        self.cq_att = CQAttention()
        self.cq_resizer = DepthwiseSeparableConv(d_model * 4, d_model, 5)
        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7)])
        self.out = Pointer()

    def forward(self, Cwid, Ccid, Qwid, Qcid):

        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()

        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)

        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)

        Ce = self.c_emb_enc(C, maskC, 1, 1)
        Qe = self.q_emb_enc(Q, maskQ, 1, 1)

        X = self.cq_att(Ce, Qe, maskC, maskQ)

        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=dropout, training=self.training)

        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0

        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0

        M0 = F.dropout(M0, p=dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0

        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2
