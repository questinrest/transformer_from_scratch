import torch
import numpy as np
import torch.nn as nn


""""
parameters to define first

 X is an input matrix, having shape (seq_len, dmodel)
 Wq is query trainable, having shape (dmodel, dk)
 Wk is key trainable, having shape (dmodel, dk)
 Wv is value trainable, having shape (dmodel, dk), dv = dk
 Wo trainable, having shape (dk*h, dmodel)
 Q = X @ Wq, shape (seq_len, dk)
 K = X @ Wk, shape (seq_len, dk)
 V  = X @ Wv, shape (seq_len, dk), dk= dv
 A = Attention(Q, K.t()), shape (seq_len, seq_len) where Attention(Q, K.t()) = (Q @ K.t())/sqrt(dk)
 Masking -- shape(seq_len, seq_len)
 then  Softmax(A), shape (seq_len, seq_len)
 second_last => Softmax(A) @ V, shape (seq_len, dk)
 lastly,  (Softmax(A) @ V) @ Wo, shape (seq_len, dmodel)
"""





class CausalAttentionSingleHead(nn.Module):
    def __init__(self, dk, dmodel, dropout):
        super(CausalAttentionSingleHead, self).__init__()
        self.dk = dk
        self.dmodel = dmodel
        self.weight_query = nn.Linear(self.dmodel, self.dk)
        self.weight_value = nn.Linear(self.dmodel, self.dk)
        self.weight_key = nn.Linear(self.dmodel, self.dk)
        ## defining Softmax
        self.softmax = nn.Softmax(dim = -1)
        ## defining dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, X):
        # calculating the Query, Key and Value tensors
        Q = self.weight_query(X)
        K = self.weight_key(X)
        V = self.weight_value(X)
        # calculating the attention score and scaling
        attn_score = (Q @ K.t())/( self.dk**0.5)
        # applying masking, first defining mask then applying
        mask = torch.triu(torch.ones((X.shape[0], X.shape[0])), diagonal=1)
        masked_attn_score = attn_score.masked_fill(mask.bool(), -torch.inf)
        # calculating the Softmax of the attn_score and adding dropouts
        A = self.dropout(self.softmax(masked_attn_score))
        # multiplying A with V
        return A @ V
    



