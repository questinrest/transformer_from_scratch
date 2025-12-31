import torch
import torch.nn as nn
import numpy as np
import math
from model.positional_encoding import PositionalEncoding
from model.transformer_block import TransformerBlock


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, h, dropout, blocks, vocab_size, seq_len):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.h = h
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.pe = PositionalEncoding(self.d_model, self.seq_len)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.linear = nn.Linear(self.d_model, self.vocab_size)
        self.dropout = dropout
        self.blocks = blocks
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        # defining transformer blocks
        self.blocks_list = nn.ModuleList()
        for i in range(self.blocks):
            self.blocks_list.append(TransformerBlock(heads = self.h, d_model= d_model, dropout=self.dropout))
        # positional encoding 
        self.positional_encoding = PositionalEncoding(d_model = self.d_model)


    # forward, assuming X (B, Seq_len, d_model) earlier, now it is X (B, Seq_len)
    def forward(self, X):
        device = X.device
        # defining embedding
        embedding = self.embedding(X)
        seq_len = X.shape[-1]
        d_model = self.d_model
        pe = self.pe(embedding)
        # modifying input tensor by adding X + positional encoding
        input_tensor = embedding + pe
        # this input will go into 6 transformer blocks
        x = input_tensor.clone()
        for block in self.blocks_list:
            y = block(x)
            x = y        
        # adding layer norm 1 
        layer_norm_1 = self.layernorm(x)

        # adding linear layer
        linear_layer_output = self.linear(layer_norm_1)
        # returning final layer output
        return linear_layer_output




