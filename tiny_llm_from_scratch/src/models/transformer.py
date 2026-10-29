import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class KVCache:
    """Key-Value cache enabling O(1) step complexity during autoregressive generation."""
    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = None
        self.v = None


def precompute_rope_freqs(
    dim: int, max_seq_len: int, theta: float = 10000.0, scale_factor: float = 1.0
) -> torch.Tensor:
    """Precomputes complex exponential frequencies for RoPE with optional linear scaling."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32) / scale_factor
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key states with start position offset."""
    T = xq.shape[1]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis[start_pos : start_pos + T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    xq_out = torch.view_as_real(xq_ * fc).flatten(3)
    xk_out = torch.view_as_real(xk_ * fc).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020)."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELUMLP(nn.Module):
    """Standard GELU feed-forward network (legacy compatibility)."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention supporting MHA, GQA, and Key-Value Caching."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.pos_emb = config.pos_emb
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        if self.pos_emb == "rope" and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis, start_pos=start_pos)

        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_head, T, head_dim)

        # Update and retrieve cached K, V
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Expand K, V for Grouped Query Attention if n_kv_head < n_head
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # PyTorch SDPA with causal masking
        dropout_p = self.dropout if self.training else 0.0
        # If single token step with cache, all keys in cache are in the past -> is_causal=False
        is_causal = (T > 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer decoder block."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.attn_norm = norm_cls(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = norm_cls(config.n_embd, eps=config.norm_eps)

        if config.ffn_type == "swiglu":
            self.mlp = SwiGLUMLP(config.n_embd, config.ffn_dim, config.dropout)
        else:
            self.mlp = GELUMLP(config.n_embd, config.ffn_dim, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis=freqs_cis, kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ModernTransformer(nn.Module):
    """
    Modern Decoder-Only Transformer (v2.1.0).
    Features: RMSNorm, RoPE with Dynamic Extrapolation, SwiGLU, KV Cache, Tied Embeddings.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_emb == "learned":
            self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.pos_embeddings = None

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        norm_cls = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.norm = norm_cls(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        self.max_seq_len = getattr(config, "max_seq_len", 1024)
        if config.pos_emb == "rope":
            freqs_cis = precompute_rope_freqs(
                config.n_embd // config.n_head,
                max(config.block_size, self.max_seq_len),
                theta=config.rope_theta,
                scale_factor=getattr(config, "rope_scaling_factor", 1.0),
            )
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        else:
            self.freqs_cis = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def create_kv_caches(self) -> List[KVCache]:
        """Creates an empty Key-Value cache for each layer."""
        return [KVCache() for _ in range(self.config.n_layer)]

    def get_freqs_cis(self, total_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.config.pos_emb != "rope":
            return None
        # Dynamic context extrapolation
        if self.freqs_cis is None or self.freqs_cis.shape[0] < total_len:
            target_len = max(total_len, self.max_seq_len)
            scale = max(1.0, total_len / self.config.block_size)
            new_freqs = precompute_rope_freqs(
                self.config.n_embd // self.config.n_head,
                target_len,
                theta=self.config.rope_theta,
                scale_factor=scale,
            ).to(device)
            self.freqs_cis = new_freqs
        return self.freqs_cis

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        max_limit = getattr(self.config, "max_seq_len", 2048)
        if start_pos + T > max_limit:
            raise ValueError(f"Sequence position {start_pos + T} exceeds max allowed capacity {max_limit}")

        x = self.tok_embeddings(idx)
        if self.pos_embeddings is not None:
            positions = torch.arange(start_pos, start_pos + T, device=idx.device)
            x = x + self.pos_embeddings(positions)

        freqs_cis = self.get_freqs_cis(start_pos + T, x.device)

        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, freqs_cis=freqs_cis, kv_cache=layer_cache, start_pos=start_pos)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def count_parameters(self) -> int:
        """Returns the total number of unique trainable parameters."""
        if self.config.tie_weights:
            unique_params = set(self.parameters())
            return sum(p.numel() for p in unique_params if p.requires_grad)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_memory_mb(self, batch_size: int = 1, seq_len: Optional[int] = None) -> Dict[str, float]:
        """
        Estimates runtime memory requirements (Section 4):
        - Weights (FP32: 4 bytes/param, FP16: 2 bytes/param)
        - KV Cache memory per forward step
        - AdamW optimizer state (8 bytes/param for moments m and v)
        """
        seq = seq_len or self.config.block_size
        n_params = self.count_parameters()

        weights_fp32_mb = (n_params * 4) / (1024 * 1024)
        weights_fp16_mb = (n_params * 2) / (1024 * 1024)

        # KV Cache: 2 (K and V) * n_layers * batch_size * n_kv_heads * seq_len * head_dim * 4 bytes
        head_dim = self.config.n_embd // self.config.n_head
        n_kv = self.config.n_kv_head or self.config.n_head
        kv_cache_bytes = 2 * self.config.n_layer * batch_size * n_kv * seq * head_dim * 4
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        # Optimizer: 2 states (m, v) * 4 bytes = 8 bytes per trainable param + gradients (4 bytes)
        optimizer_mb = (n_params * 12) / (1024 * 1024)

        return {
            "parameters_count": n_params,
            "weights_fp32_mb": round(weights_fp32_mb, 2),
            "weights_fp16_mb": round(weights_fp16_mb, 2),
            "kv_cache_mb": round(kv_cache_mb, 2),
            "training_adamw_overhead_mb": round(optimizer_mb, 2),
            "total_inference_fp32_mb": round(weights_fp32_mb + kv_cache_mb, 2),
            "total_training_fp32_mb": round(weights_fp32_mb + optimizer_mb + (kv_cache_mb * 2), 2),
        }
