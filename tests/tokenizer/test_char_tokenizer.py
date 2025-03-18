import tempfile

import pytest
import torch

from tiny_lm.tokenizer import CharacterTokenizer


@pytest.fixture
def vocab():
    return ["a", "b", "c", "d", "e"]


@pytest.fixture
def tokenizer(vocab):
    return CharacterTokenizer(vocab=vocab)


@pytest.fixture
def sample_text():
    return "abcde"


@pytest.fixture
def unseen_text():
    return "abcdef"


def test_initialization(vocab):
    # Test initialization with predefined vocabulary
    tokenizer = CharacterTokenizer(vocab=vocab)
    assert len(tokenizer.token_to_id) == len(vocab) + 4  # +4 for special tokens

    # Test initialization without vocabulary
    empty_tokenizer = CharacterTokenizer()
    assert len(empty_tokenizer.token_to_id) == 4  # Only special tokens


def test_add_token():
    tokenizer = CharacterTokenizer()
    initial_size = len(tokenizer)

    # Add new token
    token_id = tokenizer.add_token("x")
    assert len(tokenizer) == initial_size + 1
    assert tokenizer.token_to_id["x"] == token_id
    assert tokenizer.id_to_token[token_id] == "x"

    # Add existing token (shouldn't increase size)
    token_id_again = tokenizer.add_token("x")
    assert len(tokenizer) == initial_size + 1
    assert token_id == token_id_again


def test_build_vocab_from_text():
    tokenizer = CharacterTokenizer()
    initial_size = len(tokenizer)

    # Build vocab from text
    text = "hello world"
    unique_chars = len(set(text))
    tokenizer.build_vocab_from_text(text)

    # Check if all unique characters are added
    assert len(tokenizer) == initial_size + unique_chars
    # Check each character in a single assertion
    assert all(char in tokenizer.token_to_id for char in text)


def test_encode(tokenizer, sample_text, unseen_text):
    # Test encoding with special tokens
    tokens = tokenizer.encode(sample_text)
    assert len(tokens) == len(sample_text) + 2  # +2 for BOS and EOS
    assert tokens[0] == tokenizer.token_to_id[tokenizer.BOS_TOKEN]
    assert tokens[-1] == tokenizer.token_to_id[tokenizer.EOS_TOKEN]

    # Test encoding without special tokens
    tokens = tokenizer.encode(sample_text, add_special_tokens=False)
    assert len(tokens) == len(sample_text)

    # Test encoding with unknown character
    tokens = tokenizer.encode(unseen_text, add_special_tokens=False)
    assert tokens[-1] == tokenizer.token_to_id[tokenizer.UNK_TOKEN]


def test_decode(tokenizer, sample_text):
    # Encode and then decode to check round-trip
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens)
    assert decoded == sample_text

    # Test with skip_special_tokens=False
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    assert tokenizer.BOS_TOKEN in decoded
    assert tokenizer.EOS_TOKEN in decoded


def test_encode_batch(tokenizer):
    texts = ["abc", "abcde", "a"]

    # Test with padding
    batch = tokenizer.encode_batch(texts, padding=True)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == len(texts)
    assert batch.shape[1] == 7  # max_length = len("abcde") + 2 special tokens

    # Test with custom max_length
    batch = tokenizer.encode_batch(texts, padding=True, max_length=4)
    assert batch.shape[1] == 4

    # Test without padding
    batch = tokenizer.encode_batch(texts, padding=False)
    # Check all rows have correct lengths in a single assertion
    expected_lengths = [len(tokenizer.encode(text)) for text in texts]
    actual_lengths = [len(row) for row in batch]
    assert actual_lengths == expected_lengths


def test_save_and_load(tokenizer, sample_text):
    # Create temporary file that will be automatically cleaned up
    with tempfile.NamedTemporaryFile() as temp:
        filepath = temp.name

        # Save the tokenizer
        tokenizer.save(filepath)

        # Load the tokenizer back
        loaded_tokenizer = CharacterTokenizer.load(filepath)

        # Check if the loaded tokenizer has the same vocabulary
        assert tokenizer.token_to_id == loaded_tokenizer.token_to_id
        assert len(tokenizer) == len(loaded_tokenizer)

        # Test encoding with the loaded tokenizer
        original_encoding = tokenizer.encode(sample_text)
        loaded_encoding = loaded_tokenizer.encode(sample_text)
        assert original_encoding == loaded_encoding


def test_load_nonexistent_file():
    # Test loading from a non-existent file
    with pytest.raises(FileNotFoundError):
        CharacterTokenizer.load("non_existent_file.pt")


def test_len(tokenizer, vocab):
    # Test the __len__ method
    assert len(tokenizer) == len(vocab) + 4  # +4 for special tokens
    assert len(tokenizer) == len(vocab) + 4  # +4 for special tokens
