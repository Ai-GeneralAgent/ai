# Modified: renamed and adapted for this project by Yuvraj Thakur.
"""
Standalone LLM Package
Built by Yuvraj Thakur
"""
from .configuration_llm import LLMConfig
from .modeling_llm import (
    LLMForCausalLM,
    LLMModel,
    LLMDecoderLayer,
    LLMAttention,
    LLMMLP,
    LLMRMSNorm,
    LLMRotaryEmbedding,
    LLMOutput,
)
from .tokenizer import LLMTokenizer
from .convert_weights import load_llm_checkpoint, adapt_state_dict

# Unified alias with ModernTransformer
try:
    from src.models.transformer import ModernTransformer
    from src.models.config import ModelConfig
except ImportError:
    try:
        from tiny_llm_from_scratch.src.models.transformer import ModernTransformer
        from tiny_llm_from_scratch.src.models.config import ModelConfig
    except Exception:
        ModernTransformer = LLMForCausalLM
        ModelConfig = LLMConfig

__all__ = [
    "LLMConfig",
    "LLMModel",
    "LLMForCausalLM",
    "LLMDecoderLayer",
    "LLMAttention",
    "LLMMLP",
    "LLMRMSNorm",
    "LLMRotaryEmbedding",
    "LLMOutput",
    "LLMTokenizer",
    "load_llm_checkpoint",
    "adapt_state_dict",
    "ModernTransformer",
    "ModelConfig",
]

