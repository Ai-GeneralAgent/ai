"""
BPE Tokenizer Module (Section 6)
Memory-safe BPE tokenizer loader, trainer, and vocabulary inspector.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from tokenizers import Tokenizer


class BPETokenizer:
    """Wrapper around HuggingFace ByteLevel BPE Tokenizer with safety checks."""

    def __init__(self, tokenizer: Tokenizer, path: Optional[Path] = None):
        self._tokenizer = tokenizer
        self.path = path

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BPETokenizer":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {p.resolve()}")
        tok = Tokenizer.from_file(str(p))
        return cls(tok, path=p)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_raw_tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def inspect_vocab(self) -> Dict[str, Any]:
        vocab = self._tokenizer.get_vocab()
        return {
            "vocab_size": len(vocab),
            "file_path": str(self.path) if self.path else None,
            "sample_tokens": list(vocab.keys())[:20],
        }


def load_tokenizer(path: Union[str, Path]) -> Tokenizer:
    """Loads raw Tokenizer from file with validation."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Tokenizer not found at: {p.resolve()}")
    return Tokenizer.from_file(str(p))

