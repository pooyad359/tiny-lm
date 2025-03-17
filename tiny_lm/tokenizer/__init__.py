from __future__ import annotations

from typing import Dict, List, Optional

import torch


class CharacterTokenizer:
    """
    A simple character-level tokenizer for a tiny language model.
    Each character is treated as one token.
    """
    
    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initialize the tokenizer with an optional predefined vocabulary.
        
        Args:
            vocab: Optional list of characters to use as vocabulary.
                  If None, the vocabulary will be built during training.
        """
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # If vocab is provided, add all characters to the vocabulary
        if vocab:
            for char in vocab:
                self.add_token(char)
                
    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]
    
    def build_vocab_from_text(self, text: str) -> None:
        for char in sorted(set(text)):
            self.add_token(char)
            
    def encode(self, text: str) -> List[int]:
        tokens = []
        for char in text:
            token_id = self.token_to_id.get(char)
            tokens.append(token_id)
            
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for token_id in token_ids:
            chars.append(self.id_to_token.get(token_id))
            
        return ''.join(chars)
    
    def encode_batch(
        self, 
        texts: List[str], 
        add_special_tokens: bool = True, 
        padding: bool = True, 
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        pass
    
    def __len__(self) -> int:
        return len(self.token_to_id)
    
    def __repr__(self) -> str:
        return f"CharacterTokenizer(vocab_size={len(self)})"
