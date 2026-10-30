from .config import ModelConfig
from .transformer import ModernTransformer, RMSNorm, CausalSelfAttention, SwiGLUMLP
from .lora import LoRALinear, apply_lora, get_lora_state_dict, load_lora_state_dict
from .checkpoint import (
    save_checkpoint,
    load_modern_checkpoint,
    inspect_checkpoint,
    compute_file_fingerprint,
)

# Unified LLM Architecture Integration
try:
    from llm.configuration_llm import LLMConfig
    from llm.modeling_llm import LLMForCausalLM
    from llm.tokenizer import LLMTokenizer
except ImportError:
    try:
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parent.parent.parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from llm.configuration_llm import LLMConfig
        from llm.modeling_llm import LLMForCausalLM
        from llm.tokenizer import LLMTokenizer
    except Exception:
        LLMConfig = None
        LLMForCausalLM = None
        LLMTokenizer = None

__all__ = [
    "ModelConfig",
    "ModernTransformer",
    "RMSNorm",
    "CausalSelfAttention",
    "SwiGLUMLP",
    "LoRALinear",
    "apply_lora",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "save_checkpoint",
    "load_modern_checkpoint",
    "inspect_checkpoint",
    "compute_file_fingerprint",
    "LLMConfig",
    "LLMForCausalLM",
    "LLMTokenizer",
]

