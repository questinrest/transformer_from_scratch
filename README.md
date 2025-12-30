# transformer_from_scratch
Reimplementation of decoder based transformer in pure PyTorch

raw text to loss
Raw text
   ↓
Tokenizer
   ↓
Token IDs (1D stream)
   ↓
Dataset slicing
   ↓
(x, y) sequences
   ↓
DataLoader batching
   ↓
(B, T) tensors
   ↓
Embedding
   ↓
(B, T, d_model)
   ↓
Transformer Decoder
   ↓
(B, T, vocab_size)
   ↓
Cross-Entropy Loss
