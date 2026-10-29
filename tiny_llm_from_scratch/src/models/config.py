from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for the Modern Decoder-Only Transformer."""
    vocab_size: int = 8000
    block_size: int = 256
    n_embd: int = 192
    n_head: int = 4
    n_kv_head: Optional[int] = None  # Grouped Query Attention (if None, equals n_head)
    n_layer: int = 3
    ffn_dim: Optional[int] = None    # Hidden dimension of FFN
    norm_type: str = "rmsnorm"       # "rmsnorm" or "layernorm"
    norm_eps: float = 1e-6
    pos_emb: str = "rope"            # "rope", "learned", or "none"
    ffn_type: str = "swiglu"         # "swiglu" or "gelu"
    attention_type: str = "mha"      # "mha" or "gqa"
    dropout: float = 0.0
    tie_weights: bool = True
    init_std: float = 0.02
    rope_theta: float = 10000.0
    max_seq_len: int = 1024          # Dynamic extended context window limit
    rope_scaling_factor: float = 1.0 # RoPE linear frequency scaling factor
    use_kv_cache: bool = True        # Fast O(1) inference with Key-Value Cache
    arch_version: str = "2.1.0"

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be a multiple of n_kv_head ({self.n_kv_head})")
        if self.ffn_dim is None:
            if self.ffn_type == "swiglu":
                # Standard LLaMA-style 2/3 * 4d dimension rounded to multiple of 32
                raw = int(2 * (4 * self.n_embd) / 3)
                self.ffn_dim = ((raw + 31) // 32) * 32
            else:
                self.ffn_dim = 4 * self.n_embd

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def preset_tiny(cls, vocab_size: int = 8000) -> "ModelConfig":
        """Ultra-lightweight configuration (~1.8M params), instant on CPU."""
        return cls(
            vocab_size=vocab_size,
            block_size=128,
            n_embd=128,
            n_head=4,
            n_layer=2,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            tie_weights=True,
        )

    @classmethod
    def preset_small(cls, vocab_size: int = 8000) -> "ModelConfig":
        """Small configuration (~3.2M params), matches existing checkpoint capacity."""
        return cls(
            vocab_size=vocab_size,
            block_size=256,
            n_embd=192,
            n_head=4,
            n_layer=3,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            tie_weights=True,
        )

    @classmethod
    def preset_standard(cls, vocab_size: int = 8000) -> "ModelConfig":
        """Standard configuration (~9.5M params), high capacity for local CPU."""
        return cls(
            vocab_size=vocab_size,
            block_size=512,
            n_embd=288,
            n_head=6,
            n_layer=4,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            tie_weights=True,
        )

