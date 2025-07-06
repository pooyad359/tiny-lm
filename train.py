import os
import torch
import numpy as np
import time

from tiny_lm.model.base import Transformer
from tiny_lm.model.config import ModelConfig


# BATCH_SIZE is the number of sequences processed in parallel.
BATCH_SIZE = 16
# BLOCK_SIZE is the maximum context length for predictions.
BLOCK_SIZE = 64
# Training iterations
MAX_ITERS = 500
# Evaluation iterations
EVAL_ITERS = 20
# Learning rate
LEARNING_RATE = 1e-3
# Gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS = 4


def get_batch(data, block_size, batch_size, device):
    """get a batch of data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + block_size + 1]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, eval_iters, block_size, batch_size, device):
    """Estimate the loss of the model on the data."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data[split], block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def main():
    """Main function to train the model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Create vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Create tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # Create data splits
    n = int(0.9 * len(text))
    train_data = np.array(encode(text[:n]), dtype=np.uint16)
    val_data = np.array(encode(text[n:]), dtype=np.uint16)
    del text

    # Create model
    model_config = ModelConfig(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.2,
    )
    model = Transformer(model_config)
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    t0 = time.time()
    for iter in range(MAX_ITERS):
        # every once in a while evaluate the loss on train and val sets
        if iter % EVAL_ITERS == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, EVAL_ITERS, BLOCK_SIZE, BATCH_SIZE, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        if (iter + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")

    with torch.no_grad():
        torch.save(model.state_dict(), "models/tiny_lm.pt")

    print(f"Training finished in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
