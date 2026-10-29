import math
from typing import Optional, Tuple
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


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes complex exponential frequencies for Rotary Position Embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key states."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
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
    """Causal Self-Attention supporting MHA and Grouped-Query Attention (GQA)."""
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

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        if self.pos_emb == "rope" and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_head, T, head_dim)

        # Expand K, V for Grouped Query Attention if n_kv_head < n_head
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # PyTorch 2.0+ Scaled Dot Product Attention with causal masking
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
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

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ModernTransformer(nn.Module):
    """
    Modern Decoder-Only Transformer (v2.0.0).
    Features: RMSNorm, RoPE, SwiGLU, GQA/MHA, Tied Embeddings, SDPA attention.
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

        if config.pos_emb == "rope":
            freqs_cis = precompute_rope_freqs(
                config.n_embd // config.n_head,
                config.block_size,
                theta=config.rope_theta,
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

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")

        x = self.tok_embeddings(idx)
        if self.pos_embeddings is not None:
            positions = torch.arange(T, device=idx.device)
            x = x + self.pos_embeddings(positions)

        freqs_cis = self.freqs_cis[:T].to(x.device) if self.freqs_cis is not None else None

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def count_parameters(self) -> int:
        """Returns the total number of unique trainable parameters."""
        if self.config.tie_weights:
            # Avoid counting tied weights twice
            unique_params = set(self.parameters())
            return sum(p.numel() for p in unique_params if p.requires_grad)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

