import torch
from torch.utils.data import Dataset
from utils.tokenizer import CharTokenizer

class CustomTextData(Dataset):
    def __init__(self, text, tokenizer,seq_len):
        self.tokenizer = tokenizer
        self.text = text
        self.seq_len = seq_len
        self.encode = self.tokenizer.encode(self.text)
        self.seq_len = seq_len


    def __len__(self):
        return len(self.text) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.LongTensor(self.encode[idx:idx+self.seq_len])
        y = torch.LongTensor(self.encode[idx+1: idx+ self.seq_len+1])
        return x, y
