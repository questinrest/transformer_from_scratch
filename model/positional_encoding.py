## implement positional encoding using torch
import torch
import math

class PositionalEncoding:
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
    
    def __sin(self, pos, i):
        return math.sin(pos/10000**(2*i/self.d_model))
    
    def __cos(self, pos, i):
        return math.cos(pos/10000**(2*i/self.d_model))
    
    def calculate_pe(self):
        positional_encoding_vector = []
        for pos in range(self.seq_len):
            temp = []
            for i in range(self.d_model//2):
                    temp.append(self.__sin(pos,i))
                    temp.append(self.__cos(pos,i))
            positional_encoding_vector.append(temp)
        return torch.tensor(positional_encoding_vector)
        
                
    