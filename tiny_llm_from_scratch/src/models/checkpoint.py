import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import ModelConfig
from .transformer import ModernTransformer


def compute_file_fingerprint(path: Union[str, Path]) -> str:
    """Computes a fast SHA-256 fingerprint for a file."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def inspect_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Inspects a checkpoint and returns structured format metadata."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p.resolve()}")

    raw = torch.load(p, map_location="cpu", weights_only=False)
    info = {
        "file_path": str(p),
        "size_bytes": p.stat().st_size,
        "format": "unknown",
        "vocab_size": None,
        "step": 0,
        "val_loss": None,
    }

    if isinstance(raw, dict):
        if "arch_version" in raw or raw.get("config", {}).get("arch_version"):
            info["format"] = "v2_modern"
            info["config"] = raw.get("config", {})
            info["vocab_size"] = info["config"].get("vocab_size")
            info["step"] = raw.get("step", 0)
            info["val_loss"] = raw.get("val_loss")
        elif "model_state_dict" in raw and "block_size" in raw and "n_embd" in raw:
            info["format"] = "v1_legacy_bpe"
            info["vocab_size"] = raw.get("vocab_size", 8000)
            info["step"] = raw.get("step", 0)
            info["val_loss"] = raw.get("val_loss")
            info["block_size"] = raw.get("block_size")
            info["n_embd"] = raw.get("n_embd")
            info["n_head"] = raw.get("n_head")
            info["n_layer"] = raw.get("n_layer")
        elif "config" in raw and "stoi" in raw:
            info["format"] = "v0_legacy_char"
            info["vocab_size"] = len(raw.get("stoi", {}))
            info["step"] = raw.get("training_steps", 0)
    return info


def save_checkpoint(
    model: ModernTransformer,
    output_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    val_loss: Optional[float] = None,
    tokenizer_path: Optional[str] = None,
    dataset_path: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically serializes model checkpoint with rich metadata and RNG state."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_path = p.with_suffix(".tmp")

    payload = {
        "arch_version": "2.0.0",
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        "tokenizer_fingerprint": compute_file_fingerprint(tokenizer_path) if tokenizer_path else None,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_fingerprint": compute_file_fingerprint(dataset_path) if dataset_path else None,
        "torch_rng_state": torch.get_rng_state(),
        "metadata": extra_metadata or {},
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()

    try:
        torch.save(payload, temp_path)
        temp_path.replace(p)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return p


def load_modern_checkpoint(
    path: Union[str, Path],
    device: str = "cpu",
    expected_tokenizer_path: Optional[str] = None,
) -> Tuple[ModernTransformer, Dict[str, Any]]:
    """Loads a v2 ModernTransformer checkpoint with strict compatibility validation."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p.resolve()}")

    payload = torch.load(p, map_location=device, weights_only=False)

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"File {p.name} is not a valid checkpoint dictionary.")

    cfg_dict = payload.get("config")
    if not cfg_dict:
        raise ValueError(f"Checkpoint {p.name} is missing model configuration.")

    config = ModelConfig.from_dict(cfg_dict)
    model = ModernTransformer(config).to(device)
    model.load_state_dict(payload["model_state_dict"])

    # Tokenizer verification if fingerprint available
    if expected_tokenizer_path and payload.get("tokenizer_fingerprint"):
        current_fp = compute_file_fingerprint(expected_tokenizer_path)
        saved_fp = payload["tokenizer_fingerprint"]
        if current_fp and saved_fp and current_fp != saved_fp:
            print(f"WARNING: Tokenizer fingerprint mismatch between checkpoint ({saved_fp}) and current ({current_fp})")

    return model, payload

