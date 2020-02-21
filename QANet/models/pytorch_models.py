import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import pdb


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Conv1D(nn.Module):
    def __init__(self, nx, nf, wbias=True):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.wbias = wbias
        if self.wbias:
            self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        if self.wbias:
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        else:
            x = torch.mm(x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()

        if dim == 1:
            self.depthwise_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k // 2,
                bias=False,
            )
            self.pointwise_conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                padding=0,
                bias=bias,
            )
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k // 2,
                bias=False,
            )
            self.pointwise_conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                padding=0,
                bias=bias,
            )
        else:
            raise Exception(" Parameter 'dim' must have value 1 or 2")

        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        if bias:
            nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):
    def __init__(self, n_layers, size, dropout, nonlinear_f=F.relu):
        super().__init__()

        self.n_layers = n_layers
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n_layers)])
        self.linear = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(self.n_layers)]
        )
        self.nonlinear_f = nonlinear_f
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(self.n_layers):
            T = torch.sigmoid(self.gate[i](x))
            H = self.nonlinear_f(self.linear[i](x))
            H = self.dropout(H)
            x = H * T + x * (1 - T)
        return x


class Embedding(nn.Module):
    def __init__(
        self,
        d_char,
        d_word,
        d_model,
        dropout,
        dropout_char,
        wproj=True,
        depthwise=True,
    ):
        super().__init__()

        self.wproj = wproj
        self.depthwise = depthwise
        self.dropout_char = nn.Dropout(dropout_char)
        self.dropout_word = nn.Dropout(dropout)

        if depthwise:
            self.conv2d = DepthwiseSeparableConv(
                in_ch=d_char, out_ch=d_model, k=5, dim=2
            )
        else:
            self.conv2d = nn.Conv2d(
                d_char, d_model, kernel_size=5, padding=0, bias=True
            )
            nn.init.kaiming_normal_(self.conv2d.weight)

        if self.wproj:
            self.proj = nn.Linear(d_word + d_model, d_model, bias=False)
            self.high = Highway(n_layers=2, size=d_model, dropout=dropout)
        else:
            self.high = Highway(n_layers=2, size=d_word + d_model, dropout=dropout)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = self.dropout_char(ch_emb)
        ch_emb = self.conv2d(ch_emb) if self.depthwise else F.relu(self.conv2d(ch_emb))
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()

        wd_emb = self.dropout_word(wd_emb)
        wd_emb = wd_emb.transpose(1, 2)

        emb = torch.cat([ch_emb, wd_emb], dim=1).transpose(2, 1)
        if self.wproj:
            emb = self.proj(emb)
        emb = self.high(emb)
        return emb


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, max_len, d_model, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)
        bsize = x.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(bsize, -1, self.n_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (x, x, x))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(bsize, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)


class FeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, n_conv_layers, n_channels, k, max_len, n_heads, dropout):
        super().__init__()

        self.p = dropout
        self.dropout = nn.Dropout(self.p)
        self.pos_encoder = PositionalEncoding(
            max_len=max_len, d_model=n_channels, dropout=dropout
        )
        self.n_conv_layers = n_conv_layers
        self.convs = nn.ModuleList(
            [
                DepthwiseSeparableConv(in_ch=n_channels, out_ch=n_channels, k=k)
                for _ in range(n_conv_layers)
            ]
        )
        self.self_attn = MultiHeadedAttention(
            n_heads=n_heads, d_model=n_channels, dropout=dropout
        )
        self.FF = FeedForward(d_model=n_channels, d_ff=n_channels, dropout=dropout)
        self.norm_1 = nn.ModuleList(
            [nn.LayerNorm(n_channels) for _ in range(n_conv_layers)]
        )
        self.norm_2 = nn.LayerNorm(n_channels)
        self.norm_3 = nn.LayerNorm(n_channels)

    def forward(self, x, mask, l, n_blocks):
        total_sublayers = (self.n_conv_layers + 2) * n_blocks
        out = self.pos_encoder(x)

        # sublayer 1: LayerNorm -> conv -> residual connection
        l = 1
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_1[i](out)
            if (i) % 2 == 0:
                out = self.dropout(out)
            out = conv(out.transpose(1, 2)).transpose(1, 2)
            out = self._layer_dropout(out, res, self.p * float(l) / total_sublayers)
            l += 1

        # sublayer 2: LayerNorm -> Attention -> residual connection
        res = out
        out = self.norm_2(out)
        out = self.dropout(out)
        out = self.self_attn(out, mask)
        out = self._layer_dropout(out, res, self.p * float(l) / total_sublayers)
        l += 1

        # sublayer 3: LayerNorm -> FeedForward -> residual connection
        res = out
        out = self.norm_3(out)
        out = self.dropout(out)
        out = self.FF(out)
        out = self._layer_dropout(out, res, self.p * float(l) / total_sublayers)

        return out

    def _layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(nn.Module):
    def __init__(self, d_model, dropout, optimized=True):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.optimized = optimized
        if self.optimized:
            w_c = torch.empty(d_model, 1)
            w_q = torch.empty(d_model, 1)
            w_mul = torch.empty(1, 1, d_model)
            bias = torch.empty(1)
            self.w_c = nn.Parameter(nn.init.xavier_uniform_(w_c))
            self.w_q = nn.Parameter(nn.init.xavier_uniform_(w_q))
            self.w_mul = nn.Parameter(nn.init.xavier_uniform_(w_mul))
            self.bias = nn.Parameter(nn.init.constant_(bias, 0))
        else:
            w = torch.empty(d_model * 3)
            self.w = nn.Parameter(nn.init.uniform_(w))

    def forward(self, C, Q, Cmask, Qmask):
        Cmask = Cmask.unsqueeze(2)
        Qmask = Qmask.unsqueeze(1)
        batch_size_c = C.size()[0]
        if self.optimized:
            S = self._optimized_trilinear(C, Q)
        else:
            S = self._trilinear(C, Q)
        S1 = F.softmax(S.masked_fill(Qmask == 0, -1e9), dim=2)
        S2 = F.softmax(S.masked_fill(Cmask == 0, -1e9), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out

    def _trilinear(self, C, Q):
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        C = self.dropout(C)
        Q = self.dropout(Q)
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)
        subres = torch.cat([Ct, Qt, CQ], dim=3)
        res = torch.matmul(subres, self.w)
        return res

    def _optimized_trilinear(self, C, Q):
        C = self.dropout(C)
        Q = self.dropout(Q)
        subres0 = torch.matmul(C, self.w_c).expand([-1, -1, Q.size(1)])
        subres1 = torch.matmul(Q, self.w_q).transpose(1, 2).expand([-1, C.size(1), -1])
        subres2 = torch.matmul(C * self.w_mul, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.w_1 = nn.Linear(d_model * 2, 1, bias=False)
        self.w_2 = nn.Linear(d_model * 2, 1, bias=False)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=2)
        X2 = torch.cat([M1, M3], dim=2)
        p1 = self.w_1(X1).squeeze(2).masked_fill(mask == 0, -1e9)
        p2 = self.w_2(X2).squeeze(2).masked_fill(mask == 0, -1e9)
        return p1, p2


class QANet(nn.Module):
    def __init__(
        self,
        word_mat=None,
        char_mat=None,
        word_vocab=None,
        char_vocab=None,
        d_word=300,
        d_char=64,
        d_model=96,
        dropout=0.1,
        dropout_char=0.05,
        para_limit=400,
        ques_limit=50,
        n_heads=1,
        freeze=True,
    ):
        super().__init__()

        # Embedding lookup tables
        if word_mat is not None:
            self.word_emb = nn.Embedding.from_pretrained(
                torch.Tensor(word_mat), freeze=freeze
            )
        else:
            self.word_emb = nn.Embedding(len(word_vocab.itos), d_word)
            # nn.init.normal_(self.word_emb.weight, std=0.1)

        if char_mat is not None:
            self.char_emb = nn.Embedding.from_pretrained(
                torch.Tensor(char_mat), freeze=False
            )
        else:
            self.char_emb = nn.Embedding(len(char_vocab.itos), d_char)
            # nn.init.normal_(self.char_emb.weight, std=0.1)

        self.dropout = nn.Dropout(dropout)
        # Input Embedding Layer
        self.emb = Embedding(d_char, d_word, d_model, dropout, dropout_char)
        # Embedding Encoder Layer
        self.c_emb_enc = EncoderBlock(
            n_conv_layers=4,
            n_channels=d_model,
            k=7,
            max_len=para_limit,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.q_emb_enc = EncoderBlock(
            n_conv_layers=4,
            n_channels=d_model,
            k=7,
            max_len=ques_limit,
            n_heads=n_heads,
            dropout=dropout,
        )
        # Context-Query Attention Layer
        self.cq_attn = CQAttention(d_model, dropout)
        self.cq_resize = DepthwiseSeparableConv(in_ch=d_model * 4, out_ch=d_model, k=5)
        # Model Encoder Layer
        self.model_enc_blks = nn.ModuleList(
            [
                EncoderBlock(
                    n_conv_layers=2,
                    n_channels=d_model,
                    k=5,
                    max_len=para_limit,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(7)
            ]
        )
        # Output layer
        self.out = Pointer(d_model)

    def forward(self, Cwid, Ccid, Qwid, Qcid):

        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()

        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)

        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)

        Ce = self.c_emb_enc(C, maskC, 1, 1)
        Qe = self.q_emb_enc(Q, maskQ, 1, 1)

        X = self.cq_attn(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resize(X.transpose(2, 1)).transpose(2, 1)
        M0 = self.dropout(M0)

        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M1 = M0

        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M2 = M0

        M0 = self.dropout(M0)
        for i, blk in enumerate(self.model_enc_blks):
            M0 = blk(M0, maskC, i * (2 + 2) + 1, 7)
        M3 = M0

        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2
