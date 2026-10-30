# Modified: renamed and adapted for this project by Yuvraj Thakur.
"""
Weight Adapter & Converter
Translates weights between:
1. Qwen HuggingFace safetensors / PyTorch checkpoints
2. Project ModernTransformer v2.2 checkpoints
3. Standalone LLMForCausalLM model state dict
"""
from pathlib import Path
from typing import Dict, Union

import torch
import sys
if __package__ is None or __package__ == "":
    pkg_dir = Path(__file__).resolve().parent
    root_dir = pkg_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from llm.configuration_llm import LLMConfig
    from llm.modeling_llm import LLMForCausalLM
else:
    from .configuration_llm import LLMConfig
    from .modeling_llm import LLMForCausalLM


def adapt_state_dict(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Translates arbitrary source keys into LLMForCausalLM naming convention."""
    new_state = {}
    for k, v in raw_state.items():
        # Strip prefixes
        clean_k = k
        if clean_k.startswith("model.model."):
            clean_k = clean_k.replace("model.model.", "model.")

        # From-scratch ModernTransformer mapping
        clean_k = clean_k.replace("tok_embeddings.weight", "model.embed_tokens.weight")
        clean_k = clean_k.replace("blocks.", "model.layers.")
        clean_k = clean_k.replace(".attn.q_proj.", ".self_attn.q_proj.")
        clean_k = clean_k.replace(".attn.k_proj.", ".self_attn.k_proj.")
        clean_k = clean_k.replace(".attn.v_proj.", ".self_attn.v_proj.")
        clean_k = clean_k.replace(".attn.out_proj.", ".self_attn.o_proj.")
        clean_k = clean_k.replace(".attn_norm.", ".input_layernorm.")
        clean_k = clean_k.replace(".mlp_norm.", ".post_attention_layernorm.")
        clean_k = clean_k.replace(".mlp.w1.", ".mlp.gate_proj.")
        clean_k = clean_k.replace(".mlp.w3.", ".mlp.up_proj.")
        clean_k = clean_k.replace(".mlp.w2.", ".mlp.down_proj.")

        new_state[clean_k] = v
    return new_state


def load_llm_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Optional[LLMConfig] = None,
    device: str = "cpu",
) -> LLMForCausalLM:
    """Loads a checkpoint into an LLMForCausalLM instance."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p.resolve()}")

    payload = torch.load(p, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        raw_state = payload["model_state_dict"]
        cfg_dict = payload.get("config", {})
    else:
        raw_state = payload
        cfg_dict = {}

    adapted = adapt_state_dict(raw_state)

    if config is None:
        if cfg_dict:
            config = LLMConfig.from_dict(cfg_dict)
        else:
            # Infer from weights
            vocab_size, hidden_size = adapted["model.embed_tokens.weight"].shape
            config = LLMConfig(vocab_size=vocab_size, hidden_size=hidden_size)

    model = LLMForCausalLM(config).to(device)
    model.load_state_dict(adapted, strict=False)
    model.eval()
    return model

