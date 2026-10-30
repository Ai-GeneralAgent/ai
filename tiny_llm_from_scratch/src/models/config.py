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
    max_seq_len: int = 2_097_152          # 2M token extended context window limit (2,097,152 tokens)
    rope_scaling_type: str = "yarn"       # "yarn", "dynamic_ntk", or "linear"
    rope_scaling_factor: float = 1.0      # RoPE frequency scaling factor
    sliding_window: Optional[int] = None  # Optional sliding window size for StreamingLLM
    sink_tokens: int = 4                  # Attention sink token count (Xiao et al., 2023)
    chunk_size: int = 512                 # Chunked prefill size for ultra-long context
    use_kv_cache: bool = True             # Fast O(1) inference with Key-Value Cache
    n_predict_tokens: int = 1             # Multi-token prediction depth (k=1: standard next token; k=2,3: extra heads)
    arch_version: str = "2.2.0"

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

    @classmethod
    def preset_1m(cls, vocab_size: int = 10000) -> "ModelConfig":
        """Configuration tuned for ultra-long 1,048,576 (1M) context with YaRN RoPE."""
        return cls(
            vocab_size=vocab_size,
            block_size=256,
            max_seq_len=1_048_576,
            n_embd=192,
            n_head=4,
            n_layer=3,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            rope_scaling_type="yarn",
            rope_theta=1_000_000.0,
            rope_scaling_factor=4096.0,
            sliding_window=4096,
            sink_tokens=4,
            chunk_size=512,
            tie_weights=True,
            arch_version="2.2.0",
        )

    @classmethod
    def preset_2m(cls, vocab_size: int = 10000) -> "ModelConfig":
        """Configuration tuned for ultra-long 2,097,152 (2M) context with YaRN RoPE."""
        return cls(
            vocab_size=vocab_size,
            block_size=256,
            max_seq_len=2_097_152,
            n_embd=192,
            n_head=4,
            n_layer=3,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            rope_scaling_type="yarn",
            rope_theta=2_000_000.0,
            rope_scaling_factor=8192.0,
            sliding_window=8192,
            sink_tokens=4,
            chunk_size=1024,
            tie_weights=True,
            arch_version="2.2.0",
        )

    @classmethod
    def preset_25m(cls, vocab_size: int = 8000) -> "ModelConfig":
        """25M parameter model configuration (8 layers, width 512, 8 heads, vocab 8000)."""
        return cls(
            vocab_size=vocab_size,
            block_size=512,
            max_seq_len=2048,
            n_embd=512,
            n_head=8,
            n_kv_head=4,
            n_layer=8,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            tie_weights=True,
            arch_version="2.2.0",
        )

    @classmethod
    def preset_foundation_2b(cls, vocab_size: int = 151936) -> "ModelConfig":
        """2B parameter scale foundation architecture (1.37B params, 24 layers, width 2048, 2M context)."""
        return cls(
            vocab_size=vocab_size,
            block_size=1024,
            max_seq_len=2_097_152,
            n_embd=2048,
            n_head=16,
            n_kv_head=4,
            n_layer=24,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            rope_scaling_type="yarn",
            rope_theta=2_000_000.0,
            rope_scaling_factor=8192.0,
            sliding_window=8192,
            sink_tokens=4,
            chunk_size=1024,
            tie_weights=True,
            arch_version="2.2.0",
        )

    preset_qwen_2b = preset_foundation_2b

    @classmethod
    def preset_qwen_5_3b(cls, vocab_size: int = 151936) -> "ModelConfig":
        """5.3B parameter scale matching Qwen 5.3B."""
        return cls(
            vocab_size=vocab_size,
            block_size=2048,
            max_seq_len=32768,
            n_embd=3584,
            n_head=28,
            n_kv_head=4,
            n_layer=28,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            rope_theta=1000000.0,
            tie_weights=False,
            arch_version="2.2.0",
        )

    @classmethod
    def preset_5b(cls, vocab_size: int = 16384) -> "ModelConfig":
        """5B parameter model configuration adapted for local 16k byte-level vocabulary."""
        return cls(
            vocab_size=vocab_size,
            block_size=2048,
            max_seq_len=32768,
            n_embd=3584,
            n_head=28,
            n_kv_head=4,
            n_layer=28,
            norm_type="rmsnorm",
            pos_emb="rope",
            ffn_type="swiglu",
            rope_theta=1000000.0,
            tie_weights=True,
            arch_version="2.2.0",
        )

