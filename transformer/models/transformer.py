import torch
import numpy as np
import pandas as pd
import torch.nn as nn

d_model = 771  # Embedding 的维度
d_ff = 2048
d_k = d_v = d_q = 64
n_layer = 1
n_heads = 8


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(
            pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(
            pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        # enc_inputs: [seq_len, d_model]
        self.pos_table = torch.FloatTensor(pos_table)

    # enc_inputs: [batch_size, seq_len, d_model]
    def forward(self, enc_inputs):
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)


# seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q, _ = seq_q.size()  # 1*64*5
    batch_size, len_k, _ = seq_k.size()
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)          # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    pad_attn_mask = torch.ones(batch_size, len_q, len_k)
    return pad_attn_mask  # 扩展成多维度


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    # Q: [batch_size, n_heads, len_q, d_k]
    def forward(self, Q, K, V, attn_mask):
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)  # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    # input_Q: [batch_size, len_q, d_model]
    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1,
                                   n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
        )                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                        # 前馈神经网络

    # enc_inputs: [batch_size, src_len, d_model]
    def forward(self, enc_inputs, enc_self_attn_mask):
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                          # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,    # enc_outputs: [batch_size, src_len, d_model],
                                               enc_self_attn_mask)                    # attn: [batch_size, n_heads, src_len, src_len]
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(5, d_model)
        self.pos_emb = PositionalEncoding(
            d_model)                               # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    # enc_inputs: [batch_size, src_len]
    def forward(self, enc_inputs):
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
        # print('1',enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs :   [batch_size, src_len, d_model],                                                                # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.projection = nn.Linear(d_model, 1, bias=False)
        # self.projection1 = nn.Linear(128, 1, bias=False)

    # enc_inputs: [batch_size, src_len]                                                   # dec_inputs: [batch_size, tgt_len]
    def forward(self, enc_inputs):
        # enc_outputs: [batch_size, src_len, d_model],
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # dec_logits: [batch_size, tgt_len, 1]
        dec_logits = self.projection(enc_outputs)
        # dec_logits = self.projection1(dec_logits)                      # dec_logits: [batch_size, tgt_len, 1]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns
