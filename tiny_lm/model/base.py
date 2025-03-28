import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking for a decoder-only transformer model.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.head_size = config.n_embd // config.n_head

    def forward(self, x):
        batch_size, seq_len, n_embd = x.size()

        # calculate query, key, values for all heads in batch
        # (B, T, 3*n_embd)
        q, k, v = self.c_attn(x).chunk(3, dim=2)

        # reshape q, k, v for multi-head attention
        # (B, T, n_head, head_size)
        q = q.view(batch_size, seq_len, self.n_head, self.head_size)
        k = k.view(batch_size, seq_len, self.n_head, self.head_size)
        v = v.view(batch_size, seq_len, self.n_head, self.head_size)

        # transpose to (B, n_head, T, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scaled dot-product attention
        # (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # apply causal mask to prevent attending to future positions
        mask = self.mask[:, :, :seq_len, :seq_len]
        att = att.masked_fill(mask == 0, float("-inf"))

        # apply softmax to get attention weights
        att = F.softmax(att, dim=-1)

        # apply dropout
        att = self.attn_dropout(att)

        # weighted aggregation of values
        # (B, n_head, T, head_size)
        y = att @ v

        # transpose and reshape back
        # (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)

        # output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))

        return y


class FeedForward(nn.Module):
    """
    Simple feed-forward network with GELU activation and residual connection.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # expand
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # project back
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # self-attention with residual connection
        x = x + self.attn(self.ln_1(x))
        # feed-forward with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x
