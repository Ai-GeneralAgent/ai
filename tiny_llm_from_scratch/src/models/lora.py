import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation Linear Layer (Hu et al., 2021).
    Wraps an existing nn.Linear, freezes base weights, and trains rank-r adapters.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Init A with kaiming uniform, B with zeros (ensures delta is 0 at initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)
        base_out = self.base_layer(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self):
        """Merges LoRA weights into base layer weights for zero-overhead inference."""
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Unmerges LoRA weights from base layer."""
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False


def apply_lora(
    model: nn.Module,
    r: int = 4,
    alpha: float = 8.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0,
) -> Dict[str, LoRALinear]:
    """Applies LoRA to specified linear modules and freezes all base model parameters."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "w1", "w3"]

    # Freeze entire base model
    for p in model.parameters():
        p.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                sub_layer = getattr(module, target)
                if isinstance(sub_layer, nn.Linear):
                    lora_wrap = LoRALinear(sub_layer, r=r, alpha=alpha, dropout=dropout)
                    setattr(module, target, lora_wrap)
                    lora_layers[f"{name}.{target}"] = lora_wrap

    return lora_layers


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts only trainable LoRA adapter weights."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, lora_weights: Dict[str, torch.Tensor]):
    """Loads saved LoRA weights into the model."""
    model.load_state_dict(lora_weights, strict=False)

