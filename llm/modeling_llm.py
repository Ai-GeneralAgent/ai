# Modified: renamed and adapted for this project by Yuvraj Thakur.
"""
Standalone LLM Neural Architecture (Pure PyTorch Implementation)
Features:
- RMSNorm with epsilon stabilization
- SwiGLU non-linear feedforward networks
- Grouped-Query Attention (GQA) with QK-normalization
- Rotary Position Embeddings (RoPE)
- PyTorch Scaled Dot-Product Attention (SDPA)
- Key-Value (KV) Cache for fast incremental decoding
- Gradient checkpointing support
- No dependencies on transformers internal modules
"""
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_llm import LLMConfig


@dataclass
class LLMOutput:
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    loss: Optional[torch.Tensor] = None


class LLMRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        return (normed * self.weight.to(torch.float32)).to(orig_dtype)


class LLMMLP(nn.Module):
    """SwiGLU Multi-Layer Perceptron."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLMRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k shape: [B, H, S, D] -> cos, sin shape: [S, D]
    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, S, D]
    sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, S, D]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class LLMAttention(nn.Module):
    """
    Grouped-Query Attention with QK-Norm and Scaled Dot-Product Attention (SDPA).
    """

    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_heads)
        self.num_kv_heads = config.num_key_value_heads or self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK-Norm for exceptional training stability at depth
        self.q_norm = LLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)

        # Apply QK-Norm
        q = self.q_norm(q).transpose(1, 2)  # [B, H, S, D]
        k = self.k_norm(k).transpose(1, 2)  # [B, KV_H, S, D]
        v = v.transpose(1, 2)               # [B, KV_H, S, D]

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV-Cache management
        present_kv = None
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        if use_cache:
            present_kv = (k, v)

        # GQA KV repetition if num_kv_groups > 1
        if self.num_kv_groups > 1:
            k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_expanded = k
            v_expanded = v

        # Scaled Dot-Product Attention (SDPA with causal masking)
        is_causal = (past_kv is None and S > 1)
        attn_out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            is_causal=is_causal,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        out = self.o_proj(attn_out)
        return out, present_kv


class LLMDecoderLayer(nn.Module):
    """Transformer Decoder Block with Pre-RMSNorm and Residual Connections."""

    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LLMAttention(config, layer_idx)
        self.mlp = LLMMLP(config)
        self.input_layernorm = LLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-Norm Attention
        normed = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(normed, cos=cos, sin=sin, past_kv=past_kv, use_cache=use_cache)
        hidden_states = hidden_states + attn_out

        # Pre-Norm MLP
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + mlp_out
        return hidden_states, present_kv


class LLMModel(nn.Module):
    """Core Transformer BackBone."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LLMDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLMRotaryEmbedding(
            dim=config.head_dim or (config.hidden_size // config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, S = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Compute position offset if past KV cache is present
        start_pos = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total_len = start_pos + S
        cos, sin = self.rotary_emb(hidden_states, seq_len=total_len)
        cos = cos[start_pos:total_len]
        sin = sin[start_pos:total_len]

        present_kvs = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Custom checkpointing
                def custom_forward(h):
                    return layer(h, cos, sin, None, False)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(custom_forward, hidden_states, use_reentrant=False)
                present_kv = None
            else:
                hidden_states, present_kv = layer(
                    hidden_states, cos=cos, sin=sin, past_kv=layer_past, use_cache=use_cache
                )

            if use_cache:
                present_kvs.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, present_kvs


class LLMForCausalLM(nn.Module):
    """Full Language Model with Causal LM Head and Text Generation Utilities."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.model = LLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        std = self.config.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> LLMOutput:
        hidden_states, present_kvs = self.model(
            input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return LLMOutput(logits=logits, past_key_values=present_kvs, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_token_ids: Optional[List[int]] = None,
        stream_callback: Optional[Callable[[int], None]] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-caching and sampling controls."""
        self.eval()
        stop_ids = set(stop_token_ids or [self.config.eos_token_id])
        curr_ids = input_ids.clone()
        past_kv = None

        for _ in range(max_new_tokens):
            if past_kv is None:
                out = self.forward(curr_ids, use_cache=True)
            else:
                out = self.forward(curr_ids[:, -1:], past_key_values=past_kv, use_cache=True)

            logits = out.logits[:, -1, :] / max(1e-5, temperature)
            past_kv = out.past_key_values

            # Repetition Penalty
            if repetition_penalty != 1.0:
                for token_id in set(curr_ids[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # Top-K
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Top-P (Nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            token_int = next_token.item()

            if stream_callback:
                stream_callback(token_int)

            if token_int in stop_ids:
                break

        return curr_ids

