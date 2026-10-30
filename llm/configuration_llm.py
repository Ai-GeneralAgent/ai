# Modified: renamed and adapted for this project by Yuvraj Thakur.
"""
LLM Configuration (Standalone Vendored Architecture)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class LLMConfig:
    """
    Configuration class for the standalone LLM architecture.
    Fully configurable from tiny edge models (3M) up to full-scale (2B+).
    """

    model_type: str = "llm"
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643

    @classmethod
    def preset_tiny(cls, vocab_size: int = 10000) -> "LLMConfig":
        """Ultra-fast 3.25M parameter model for fast CPU training and testing."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=192,
            intermediate_size=512,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=48,
            max_position_embeddings=2048,
            tie_word_embeddings=True,
        )

    @classmethod
    def preset_25m(cls, vocab_size: int = 8000) -> "LLMConfig":
        """25M parameter model configuration (8 layers, width 512, 8 heads, vocab 8000)."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=512,
            intermediate_size=1408,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            max_position_embeddings=2048,
            tie_word_embeddings=True,
        )

    @classmethod
    def preset_small(cls, vocab_size: int = 10000) -> "LLMConfig":
        """Fast 28M parameter model for single PC experimentation."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            max_position_embeddings=4096,
            tie_word_embeddings=True,
        )

    @classmethod
    def preset_qwen_2b(cls, vocab_size: int = 151936) -> "LLMConfig":
        """Exact architecture configuration matching 2B parameter scale."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            max_position_embeddings=32768,
            tie_word_embeddings=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

