from dataclasses import dataclass


@dataclass
class Parameters:
    device : str | None = None
    epochs : int | None = None
    learning_rate : float | None = None
    d_model : int | None = None
    h : int | None = None
    blocks : int | None = None
    vocab_size : int | None = None
    dropout : float | None = None
    batch_size : int | None = None