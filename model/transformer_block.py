import torch
import torch.nn as nn
import numpy as np
import math


class TransformerBlock(nn.Module):
    def __init__(self, heads, d_model, dropout):
        super(TransformerBlock, self).__init__()
        self.heads = heads
        self.d_model = d_model
        assert self.d_model % self.heads == 0, "Choose Correct Head Number"
        self.dk = self.d_model // self.heads
        # defining dropout
        self.dropout = nn.Dropout(dropout)
        # defining softmax
        self.softmax = nn.Softmax(dim=-1)
        # query_weight, key_weight, value_weight
        self.weight_query = nn.Linear(self.d_model, self.d_model)
        self.weight_key = nn.Linear(self.d_model, self.d_model)
        self.weight_value = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        self.ffn1 = nn.Linear(self.d_model, self.d_model*4)
        self.ffn2 = nn.Linear(self.d_model*4, self.d_model)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, X):
        # extract shape of seq_len
        seq_len = X.shape[1]
        batch_size = X.shape[0]

        ##layer normalization
        layer_norm1_output = self.layer_norm_1(X)

        #project X (B, SEQ_LEN, Dmodel) into (B, Dmodel, Dmodel)
        Q = self.weight_query(layer_norm1_output)
        K = self.weight_key(layer_norm1_output)
        V = self.weight_value(layer_norm1_output)
        Q_heads = torch.permute(torch.reshape(Q, shape = (batch_size, seq_len, self.heads, self.dk)), dims = (0, 2, 1,3))
        K_heads = torch.permute(torch.reshape(K, shape = (batch_size, seq_len, self.heads, self.dk)), dims = (0, 2, 1,3))
        V_heads = torch.permute(torch.reshape(V, shape = (batch_size, seq_len, self.heads, self.dk)), dims = (0, 2, 1,3))

        # calculate attn score and scale
        attn_score = (torch.matmul(Q_heads, K_heads.transpose(-1, -2)))/(math.sqrt(self.dk))

        # masking 
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        # applying mask
        attn_score_with_mask = attn_score.masked_fill(mask.bool(), -torch.inf)
        # applying softmax
        attn_weight_with_mask = self.softmax(attn_score_with_mask)
        # apply dropouts 
        attn_wgts_mask_drpt = self.dropout(attn_weight_with_mask)

        ### multiply with V_heads
        A = torch.matmul(attn_wgts_mask_drpt, V_heads)
        ## concatenation
        concat_A = torch.reshape(torch.permute(A, dims = (0,2,1,3)), shape = (batch_size, seq_len, self.d_model))
        output_masked_attn = self.w_o(concat_A)
        #return layer_norm1_output

        # residual connection 1
        residual_connection_1 = X + output_masked_attn

        #layer norm 2
        layer_norm_2_output =self.layer_norm_2(residual_connection_1)
        # ffnn
        ffn1 = self.ffn1(layer_norm_2_output)
        # activation
        activation = nn.ReLU(ffn1)
        # final layer
        ffn2 = self.ffn2(activation)
        
        # residual connection 2
        final_output = layer_norm_2_output + ffn2

        #return the fnal vector
        return final_output