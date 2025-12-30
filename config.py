from dataclasses import dataclass


@dataclass
class Parameters:
    device : str
    epochs : str
    learning_rate : float
    d_model : int
    h : int
    blocks : int
    vocab_size : int
    dropout : float