"""here will come dataset class and dataloader for the custom data"""
""" use this class to prepare dataset
create one column for input (seq_len) [0:seq_len]
and next column for prediction [seq_len:seq_len+1]"""


import torch
from torch.utils.data import Dataset
from utils.tokenizer import CharTokenizer

class CustomTextData(Dataset):
    def __init__(self,file_path, seq_len):
        self.file_path = file_path
        with open(self.file_path) as f:
            self.file = f.read()
        self.full_encode = CharTokenizer(self.file).encode(self.file)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.file) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.LongTensor(self.full_encode[idx:idx+self.seq_len])
        y = torch.LongTensor(self.full_encode[idx+1: idx+ self.seq_len+1])
        return x, y
