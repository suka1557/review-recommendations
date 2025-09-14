from tokenizers import ByteLevelBPETokenizer
from typing import List
from pathlib import Path

class ReviewTokenizer:

    def __init__(self, vocab_size: int, special_tokens: List):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<START>', '<END>', '<PAD>', '<UNK>']
        self.tokenizer = None

    def train(self, file_path: Path|str):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(
            files=file_path,
            vocab_size = self.vocab_size,
            special_tokens = self.special_tokens,
        )
        print(f"Tokenizer trained with vocab size = {self.vocab_size}")

    def save_tokenizer(self, save_dir: Path|str):
        """
        Save the tokenizer model files (vocab.json, merges.txt).
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been trained yet.")
        self.tokenizer.save_model(save_dir)
        print(f"Tokenizer saved at {save_dir}")

    def load_tokenizer(self, save_dir: Path|str):
        """
        Load tokenizer from saved model files.
        """
        self.tokenizer = ByteLevelBPETokenizer(
            f"{save_dir}/vocab.json",
            f"{save_dir}/merges.txt"
        )
        print(f"Tokenizer loaded from {save_dir}")

    def encode(self, text):
        """
        Encode a string into tokens and IDs.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Train or load first.")
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        """
        Decode a sequence of IDs back into text.
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Train or load first.")
        return self.tokenizer.decode(token_ids)