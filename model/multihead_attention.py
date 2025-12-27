import torch
import torch.nn as nn
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, h, dmodel, dropout):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dmodel = dmodel
        assert self.dmodel % self.h == 0, "head and dmodel configuration failed"
        self.dk = self.dmodel // self.h
        self.weight_query = nn.Linear(dmodel, dmodel)
        self.weight_key = nn.Linear(dmodel, dmodel)
        self.weight_value = nn.Linear(dmodel, dmodel)
        self.w_o = nn.Linear(dmodel, dmodel)
        # softmax
        self.softmax = nn.Softmax(dim = -1)
        # drop out
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        seq_len = X.shape[0]
        Q = self.weight_query(X)
        K = self.weight_key(X)
        V = self.weight_value(X)
        Q_head = torch.permute(torch.reshape(Q, shape=(seq_len, self.h, self.dk)), dims= (1, 0, 2))
        K_head = torch.permute(torch.reshape(K, shape=(seq_len, self.h, self.dk)), dims= (1, 0, 2))
        V_head = torch.permute(torch.reshape(V, shape=(seq_len, self.h, self.dk)), dims= (1, 0, 2))
        # calculating attention score and scaling
        attn_score = (torch.matmul(Q_head, K_head.transpose(-1,-2)))/(math.sqrt(self.dk))
        # applying masking
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        attn_score_masked = attn_score.masked_fill(mask.bool(), -torch.inf)
        # applying softmax
        attn_weights = self.softmax(attn_score_masked)
        # applying dropout
        attn_weights_with_dropout = self.dropout(attn_weights)
        # multiplying with V
        A = torch.matmul(attn_weights_with_dropout, V_head)
        # concatenation of the vector
        concate_heads = torch.reshape(torch.permute(A, dims = (1, 0, 2)), shape = (seq_len, self.dmodel))
        return self.w_o(concate_heads)
    