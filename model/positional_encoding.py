import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        positional_encoding_vector = []

        for pos in range(max_seq_len):
            temp = []
            for i in range(d_model // 2):
                temp.append(math.sin(pos / (10000 ** (2 * i / d_model))))
                temp.append(math.cos(pos / (10000 ** (2 * i / d_model))))
            positional_encoding_vector.append(temp)

        self.pe = torch.tensor(positional_encoding_vector, dtype=torch.float32)

        self.register_buffer("pe", self.pe)

    def forward(self, X):
        """
        X: (B, T, d_model)
        """
        seq_len = X.size(-2)
        return X + self.pe[:seq_len]
