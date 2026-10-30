# Modified: renamed and adapted for this project by Yuvraj Thakur.
"""
LLM Tokenizer Wrapper & Chat Template Processor
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer


class LLMTokenizer:
    """
    Tokenizer wrapper with chat template formatting and local file loading.
    Supports standard ChatML / OpenAI message formats.
    """

    SYSTEM_PREFIX = "<|im_start|>system\n"
    USER_PREFIX = "<|im_start|>user\n"
    ASSISTANT_PREFIX = "<|im_start|>assistant\n"
    END_TOKEN = "<|im_end|>\n"

    def __init__(self, tokenizer_file: Union[str, Path]):
        p = Path(tokenizer_file)
        if not p.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {p.resolve()}")
        self._tokenizer = Tokenizer.from_file(str(p))
        self.vocab_size = self._tokenizer.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        eos = self._tokenizer.token_to_id("<|im_end|>") or self._tokenizer.token_to_id("<|endoftext|>")
        return eos if eos is not None else 0

    @property
    def pad_token_id(self) -> int:
        pad = self._tokenizer.token_to_id("<|pad|>") or self.eos_token_id
        return pad

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Formats multi-turn messages into standard ChatML conversation prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "system":
                formatted.append(f"{self.SYSTEM_PREFIX}{content}{self.END_TOKEN}")
            elif role == "user":
                formatted.append(f"{self.USER_PREFIX}{content}{self.END_TOKEN}")
            elif role == "assistant":
                formatted.append(f"{self.ASSISTANT_PREFIX}{content}{self.END_TOKEN}")

        if add_generation_prompt:
            formatted.append(self.ASSISTANT_PREFIX)

        return "".join(formatted)

