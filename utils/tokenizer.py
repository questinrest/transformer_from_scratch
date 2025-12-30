import torch

class CharTokenizer:
    def __init__(self, file):
        """initializes with creating stoi and itos"""
        self.chars = []
        self.stoi = {}
        for char in file:
            self.chars.append(char)
        self.chars = sorted(set(self.chars))


        ## building string_to_index
        counter = 0
        for char in self.chars:
            if char not in self.stoi:
                self.stoi[char] = counter
                counter += 1
        ## building index to string
        self.itos = {v:k for k,v in self.stoi.items()}

    
    def encode(self, text):
        """encode given text into ids"""
        # reading text
        encoded_list = []
        for char in list(text):
            encoded_list.append(self.stoi[char])
        return encoded_list
    
    def decode(self, id_list):
        """decodes back to string given index"""
        text = ""
        for idx in id_list:
            text += self.itos[idx]
        return text
