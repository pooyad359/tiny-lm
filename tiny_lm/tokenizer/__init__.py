import os
from typing import Dict, List, Optional

import torch


class CharacterTokenizer:
    """
    A simple character-level tokenizer for a tiny language model.
    Each character is treated as one token.

    Special tokens:
    - PAD_TOKEN: Used for padding sequences to the same length
    - UNK_TOKEN: Used for characters not seen during training
    - BOS_TOKEN: Beginning of sequence marker
    - EOS_TOKEN: End of sequence marker
    """

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initialize the tokenizer with an optional predefined vocabulary.

        Args:
            vocab: Optional list of characters to use as vocabulary.
                  If None, the vocabulary will be built during training.
        """
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Initialize with special tokens
        self._add_special_tokens()

        # If vocab is provided, add all characters to the vocabulary
        if vocab:
            for char in vocab:
                self.add_token(char)

    def _add_special_tokens(self):
        """Add special tokens to the vocabulary."""
        for token in [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary if it doesn't exist.

        Args:
            token: Character to add

        Returns:
            The ID of the token
        """
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def build_vocab_from_text(self, text: str) -> None:
        """
        Build vocabulary from text.

        Args:
            text: Text to build vocabulary from
        """
        # Add each unique character to the vocabulary
        for char in sorted(set(text)):
            self.add_token(char)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Text to tokenize
            add_special_tokens: Whether to add BOS and EOS tokens

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.token_to_id[self.BOS_TOKEN])

        for char in text:
            # Use UNK for characters not in vocabulary
            token_id = self.token_to_id.get(char, self.token_to_id[self.UNK_TOKEN])
            tokens.append(token_id)

        if add_special_tokens:
            tokens.append(self.token_to_id[self.EOS_TOKEN])

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in the output

        Returns:
            Decoded text
        """
        special_tokens = {
            self.token_to_id[self.PAD_TOKEN],
            self.token_to_id[self.UNK_TOKEN],
            self.token_to_id[self.BOS_TOKEN],
            self.token_to_id[self.EOS_TOKEN]
        }

        chars = []
        for token_id in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens and token_id in special_tokens:
                continue

            chars.append(self.id_to_token.get(token_id, self.UNK_TOKEN))

        return ''.join(chars)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True,
                     padding: bool = True, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add BOS and EOS tokens
            padding: Whether to pad sequences to the same length
            max_length: Maximum length of sequences (if None, use the longest sequence)

        Returns:
            Tensor of token IDs with shape (batch_size, seq_length)
        """
        batch_tokens = [self.encode(text, add_special_tokens) for text in texts]

        # Determine max length for padding
        if max_length is None:
            max_length = max(len(tokens) for tokens in batch_tokens)

        # Pad sequences if needed
        if padding:
            pad_id = self.token_to_id[self.PAD_TOKEN]
            padded_batch = []

            for tokens in batch_tokens:
                # Truncate if longer than max_length
                if len(tokens) > max_length:
                    padded_batch.append(tokens[:max_length])
                else:
                    # Pad with PAD_TOKEN if shorter
                    padded_tokens = tokens + [pad_id] * (max_length - len(tokens))
                    padded_batch.append(padded_tokens)

            batch_tokens = padded_batch

        return torch.tensor(batch_tokens)

    def save(self, filepath: str) -> None:
        """
        Save the tokenizer vocabulary to disk.

        Args:
            filepath: Path to save the vocabulary
        """
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()}
        }
        torch.save(vocab_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'CharacterTokenizer':
        """
        Load a tokenizer vocabulary from disk.

        Args:
            filepath: Path to load the vocabulary from

        Returns:
            Loaded tokenizer
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file {filepath} not found")

        vocab_data = torch.load(filepath)

        tokenizer = cls()
        tokenizer.token_to_id = vocab_data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}

        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)
