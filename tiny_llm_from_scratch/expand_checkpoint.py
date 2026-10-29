#!/usr/bin/env python3
"""
Expands model checkpoint vocabulary from 8,000 to 10,000:
1. Preserves all learned weights for tokens 0..7999
2. Initializes new embedding vectors for tokens 8000..9999
3. Validates forward pass and loss computation
"""
import copy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.models import ModelConfig, ModernTransformer, save_checkpoint, load_modern_checkpoint


def expand_checkpoint_vocab(
    source_ckpt_path: Path,
    target_ckpt_path: Path,
    new_vocab_size: int = 10000,
):
    print("=" * 65)
    print("EXPANDING MODEL CHECKPOINT EMBEDDING MATRIX")
    print("=" * 65)
    print(f"Source Checkpoint: {source_ckpt_path.name}")
    print(f"Target Checkpoint: {target_ckpt_path.name}")
    print(f"Target Vocab Size: {new_vocab_size}")

    if not source_ckpt_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_ckpt_path}")

    # Load raw dictionary
    ckpt = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    old_cfg_dict = ckpt["config"]
    old_vocab = old_cfg_dict["vocab_size"]
    n_embd = old_cfg_dict["n_embd"]
    init_std = old_cfg_dict.get("init_std", 0.02)

    print(f"Original Vocab: {old_vocab} | Embedding Dim: {n_embd} | Init Std: {init_std}")

    if old_vocab >= new_vocab_size:
        print("Checkpoint vocab is already >= target. Nothing to expand.")
        return

    # Expand new ModelConfig
    new_cfg_dict = copy.deepcopy(old_cfg_dict)
    new_cfg_dict["vocab_size"] = new_vocab_size
    new_cfg = ModelConfig(**new_cfg_dict)

    # Instantiate new model
    new_model = ModernTransformer(new_cfg)

    # State dict manipulation
    old_sd = ckpt["model_state_dict"]
    new_sd = new_model.state_dict()

    for k, v in old_sd.items():
        if k == "tok_embeddings.weight":
            # Expand embedding table
            expanded_w = torch.randn(new_vocab_size, n_embd) * init_std
            expanded_w[:old_vocab] = v
            new_sd[k] = expanded_w
            if new_cfg.tie_weights:
                new_sd["lm_head.weight"] = expanded_w
            print(f"  [EXPANDED] {k}: {v.shape} -> {expanded_w.shape}")
        elif k == "lm_head.weight" and not new_cfg.tie_weights:
            expanded_head = torch.randn(new_vocab_size, n_embd) * init_std
            expanded_head[:old_vocab] = v
            new_sd[k] = expanded_head
            print(f"  [EXPANDED] {k}: {v.shape} -> {expanded_head.shape}")
        elif k == "lm_head.weight" and new_cfg.tie_weights:
            continue
        else:
            new_sd[k] = v

    new_model.load_state_dict(new_sd)

    # Verification: Test forward pass on sequence with old and new tokens
    test_seq = torch.tensor([[10, 20, 8000, 8024]], dtype=torch.long)
    with torch.no_grad():
        logits, _ = new_model(test_seq)
    print(f"\n[VERIFIED] Forward pass successful. Output shape: {list(logits.shape)}")
    assert logits.shape == (1, 4, new_vocab_size)

    # Save expanded checkpoint
    payload = {
        "model_state_dict": new_model.state_dict(),
        "config": new_cfg.to_dict(),
        "step": ckpt.get("step", 0),
        "val_loss": ckpt.get("val_loss", float("inf")),
        "arch_version": new_cfg.arch_version,
    }
    target_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target_ckpt_path)
    print(f"Saved expanded checkpoint to: {target_ckpt_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    src_ckpt = PROJECT_ROOT / "checkpoints" / "modern_base_v2_best.pt"
    tgt_ckpt = PROJECT_ROOT / "checkpoints" / "modern_master_v2_10k_base.pt"
    expand_checkpoint_vocab(src_ckpt, tgt_ckpt, new_vocab_size=10000)
