## implement positional encoding using torch
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
    
    def __sin(self, pos, i):
        return math.sin(pos/10000**(2*i/self.d_model))
    
    def __cos(self, pos, i):
        return math.cos(pos/10000**(2*i/self.d_model))
    
    def calculate_pe(self, X):
        seq_len = X.shape[1]
        positional_encoding_vector = []
        for pos in range(seq_len):
            temp = []
            for i in range(self.d_model//2):
                    temp.append(self.__sin(pos,i))
                    temp.append(self.__cos(pos,i))
            positional_encoding_vector.append(temp)
        return torch.tensor(positional_encoding_vector)
        
                
    