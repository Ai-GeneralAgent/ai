#!/usr/bin/env python3
"""
Optimizer Comparison Suite (AdamW vs Muon).
Evaluates AdamW (lr: 3e-4, 1e-3, 3e-3) vs Muon (muon-lr: 0.01, 0.02, 0.04)
Across 2 random seeds (42, 1337) on the real corpus with fixed validation windows.
Generates a markdown comparison table.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NESTED_ROOT = PROJECT_ROOT / "tiny_llm_from_scratch"

for p in (PROJECT_ROOT, NESTED_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from src.models.config import ModelConfig
from src.models.transformer import ModernTransformer
from src.training.trainer import Trainer, TrainingConfig
from src.data.token_cache import CompactTokenCache


def run_single_experiment(
    optimizer_name: str,
    lr: float,
    muon_lr: float,
    seed: int,
    steps: int,
    cache: CompactTokenCache,
    tok_path: Path,
) -> float:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=8000, block_size=128, n_embd=128, n_head=4, n_layer=3)
    model = ModernTransformer(cfg)
    tmp_dir = Path(tempfile.mkdtemp())
    t_cfg = TrainingConfig(
        max_steps=steps,
        learning_rate=lr,
        muon_lr=muon_lr,
        optimizer=optimizer_name,
        seed=seed,
        batch_size=4,
        grad_accum_steps=1,
        eval_interval=steps,
        output_dir=str(tmp_dir),
        device="cpu",
    )
    trainer = Trainer(model, t_cfg, cache, tok_path)
    trainer.train()
    val_loss = trainer.evaluate()
    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Compare AdamW vs Muon")
    parser.add_argument("--steps", type=int, default=500, help="Steps per experiment (e.g. 500-2000)")
    parser.add_argument("--data", default="data_knowledge_corpus.bpe_tokens.u16", help="Path to token cache")
    args = parser.parse_args()

    cache_path = NESTED_ROOT / args.data
    if not cache_path.is_file():
        cache_path = PROJECT_ROOT / args.data

    tok_path = NESTED_ROOT / "tokenizer_bpe.json"
    if not tok_path.is_file():
        tok_path = PROJECT_ROOT / "tokenizer_bpe.json"

    token_count = cache_path.stat().st_size // 2
    cache = CompactTokenCache(cache_path, token_count)

    configs = [
        {"opt": "adamw", "lr": 3e-4, "muon_lr": 0.0, "label": "AdamW (lr=3e-4)"},
        {"opt": "adamw", "lr": 1e-3, "muon_lr": 0.0, "label": "AdamW (lr=1e-3)"},
        {"opt": "adamw", "lr": 3e-3, "muon_lr": 0.0, "label": "AdamW (lr=3e-3)"},
        {"opt": "muon", "lr": 1e-3, "muon_lr": 0.01, "label": "Muon (muon_lr=0.01, lr=1e-3)"},
        {"opt": "muon", "lr": 1e-3, "muon_lr": 0.02, "label": "Muon (muon_lr=0.02, lr=1e-3)"},
        {"opt": "muon", "lr": 1e-3, "muon_lr": 0.04, "label": "Muon (muon_lr=0.04, lr=1e-3)"},
    ]

    seeds = [42, 1337]
    results = []

    print(f"\n========================================================")
    print(f"STARTING OPTIMIZER BENCHMARK: {len(configs)} configs x {len(seeds)} seeds ({args.steps} steps each)")
    print(f"========================================================\n")

    for c in configs:
        row = {"label": c["label"], "opt": c["opt"], "lr": c["lr"], "muon_lr": c["muon_lr"]}
        losses = []
        for s in seeds:
            print(f"Running {c['label']} [Seed {s}, {args.steps} steps]...")
            t0 = time.time()
            v_loss = run_single_experiment(c["opt"], c["lr"], c["muon_lr"], s, args.steps, cache, tok_path)
            el = time.time() - t0
            print(f"  -> Seed {s} Val Loss: {v_loss:.4f} in {el:.1f}s")
            row[f"val_seed_{s}"] = v_loss
            losses.append(v_loss)
        row["avg_val_loss"] = sum(losses) / len(losses)
        results.append(row)

    cache.close()

    # Print results table
    print("\n" + "=" * 70)
    print("OPTIMIZER BENCHMARK RESULTS (Fixed Deterministic Validation Loss)")
    print("=" * 70)
    print(f"{'Optimizer Configuration':<32} | {'Seed 42':<10} | {'Seed 1337':<10} | {'Average':<10}")
    print("-" * 70)
    best_row = min(results, key=lambda x: x["avg_val_loss"])
    for r in results:
        marker = " [BEST]" if r == best_row else ""
        print(f"{r['label']:<32} | {r['val_seed_42']:<10.4f} | {r['val_seed_1337']:<10.4f} | {r['avg_val_loss']:<10.4f}{marker}")
    print("=" * 70 + "\n")

    # Save to JSON
    out_file = PROJECT_ROOT / "eval" / "optimizer_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark data to: {out_file}")


if __name__ == "__main__":
    main()
