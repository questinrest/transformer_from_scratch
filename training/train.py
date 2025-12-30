import torch
import torch.nn as nn
from training.dataset import CustomTextData
from utils.tokenizer import CharTokenizer
from torch.utils.data import DataLoader
from model.decoder import TransformerDecoder



## reading the entire file

with open(r"C:\Users\amanm\Desktop\learning\transformer_from_scratch\data\tiny-shakespeare.txt") as f:
    full_text = f.read()


N = len(full_text)


## Splitting train and validation and data split

train_text = full_text[:int((0.9)*N)]
val_text = full_text[int((0.9)*N):]


# preparing tokenizer
tokenizer = CharTokenizer(train_text)

train_custom = CustomTextData(text=train_text, tokenizer=tokenizer, seq_len=20)
val_custom = CustomTextData(text=val_text, tokenizer=tokenizer, seq_len=20)


train_loader = DataLoader(train_custom, batch_size=2, shuffle = True)
val_loader = DataLoader(val_custom, batch_size=2, shuffle = False)


# --------------------------------
# 1. Device
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)

# --------------------------------
# 2. Hyperparameters
# --------------------------------
epochs = 10
learning_rate = 1e-3
d_model = 512

# --------------------------------
# 3. Vocabulary size
# --------------------------------
vocab_size = len(tokenizer.stoi)

# --------------------------------
# 4. Model
# --------------------------------
model = TransformerDecoder(
    d_model=d_model,
    h=8,
    dropout=0.1,
    blocks=6,
    vocab_size=vocab_size
).to(device)

# 🔍 Sanity check
print("Model device:", next(model.parameters()).device, flush=True)

# --------------------------------
# 5. Loss & Optimizer
# --------------------------------
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# --------------------------------
# 6. Training loop
# --------------------------------
import time

model.train()

for epoch in range(epochs):
    total_loss = 0.0
    num_batches = 0
    epoch_start = time.time()

    for x, y in train_loader:
        batch_start = time.time()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)

        loss = loss_fn(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 🔴 PRINT EVERY BATCH
        print(
            f"[Epoch {epoch+1}] "
            f"Batch {num_batches}/{len(train_loader)} | "
            f"Loss {loss.item():.4f} | "
            f"Batch time {time.time() - batch_start:.2f}s | "
            f"Epoch time {time.time() - epoch_start:.1f}s",
            flush=True
        )

    avg_loss = total_loss / max(1, num_batches)
    print(
        f"Epoch [{epoch+1}/{epochs}] DONE | "
        f"Avg Loss {avg_loss:.4f} | "
        f"Total Epoch Time {time.time() - epoch_start:.1f}s",
        flush=True
    )

