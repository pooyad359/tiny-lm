import torch

from tiny_lm.model.base import Transformer
from tiny_lm.model.config import ModelConfig


# BATCH_SIZE is the number of sequences processed in parallel.
BATCH_SIZE = 16
# BLOCK_SIZE is the maximum context length for predictions.
BLOCK_SIZE = 64


def main():
    """Main function to generate text from the model."""
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
    model.load_state_dict(torch.load("models/tiny_lm.pt"))
    model.to(device)
    model.eval()

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()
