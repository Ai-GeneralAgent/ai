#!/usr/bin/env python3
"""
Modern SFT (Supervised Fine-Tuning) Runner for Tiny LLM
Trains model on instruction/chat datasets with:
- Completion-only loss masking (-100 for user prompts)
- Anti-catastrophic forgetting replay buffer
- Loss spike & NaN protection
- Decoupled AdamW optimization
"""
import argparse
import os
from pathlib import Path
import sys
import time

# Guard OpenBLAS / MKL threads on Windows CPU (Qualcomm Oryon 8 cores)
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

# Safe UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer
from src.models import ModelConfig, ModernTransformer, load_modern_checkpoint
from src.training import InstructionDataset, finetune_instructions
from src.data import CompactTokenCache
from src.generation import TextGenerator, SamplingConfig


def main():
    parser = argparse.ArgumentParser(description="Run SFT Instruction Tuning on Modern Tiny LLM")
    parser.add_argument("--base-checkpoint", default=None, help="Base model checkpoint to fine-tune from")
    parser.add_argument("--data", default="training_data/qa_and_database_sft.jsonl", help="Path to instruction JSONL dataset")
    parser.add_argument("--replay-cache", default="data_master_v3_unified.bpe_tokens.u16", help="Optional replay corpus cache")
    parser.add_argument("--replay-ratio", type=float, default=0.20, help="Fraction of batch sampled from base replay")
    parser.add_argument("--steps", type=int, default=10000, help="Number of SFT steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-time", type=float, default=None, help="Maximum training time in seconds (e.g. 7200 for 2 hours)")
    parser.add_argument("--save-interval", type=int, default=250, help="Checkpoint saving step interval")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="muon", help="Optimizer choice (default: muon)")
    parser.add_argument("--muon-lr", type=float, default=0.01, help="Learning rate for Muon 2D orthogonal updates")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation micro-steps (effective batch = batch_size * grad_accum_steps)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps for linear warmup cosine decay schedule")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Minimum learning rate ratio for cosine decay")
    parser.add_argument("--output", default="checkpoints/modern_chat_finetuned.pt", help="Output checkpoint file")
    parser.add_argument("--device", default="auto", help="Compute device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    # Device auto-detection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[DEVICE] Compute device: {device.upper()}")

    if device == "cpu":
        try:
            torch.set_num_threads(8)
            torch.set_num_interop_threads(2)
            print(f"[THREADS] Configured PyTorch CPU parallelism: {torch.get_num_threads()} compute threads")
        except Exception:
            pass

    tok_path = PROJECT_ROOT / "tokenizer_bpe.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Resolve Base Checkpoint
    ckpt_path = None
    if args.base_checkpoint:
        ckpt_path = Path(args.base_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path

    if ckpt_path is None or not ckpt_path.is_file():
        candidates = [
            PROJECT_ROOT / "checkpoints" / "modern_master_v3_best.pt",
            PROJECT_ROOT / "checkpoints" / "modern_master_20min_best.pt",
            PROJECT_ROOT / "checkpoints" / "modern_production_v2_best.pt",
            PROJECT_ROOT / "checkpoints" / "modern_base_v2_best.pt",
            PROJECT_ROOT / "checkpoints" / "modern_knowledge_v2_best.pt",
        ]
        for c in candidates:
            if c.is_file():
                ckpt_path = c
                break

    if ckpt_path and ckpt_path.is_file():
        print(f"[MODEL] Loading base checkpoint: {ckpt_path.name}")
        model, payload = load_modern_checkpoint(ckpt_path, device=device)
    else:
        print("[MODEL] Initializing fresh small architecture...")
        cfg = ModelConfig.preset_small(vocab_size=tokenizer.get_vocab_size())
        model = ModernTransformer(cfg).to(device)

    # Optional Replay Cache
    replay_cache = None
    replay_p = Path(args.replay_cache)
    if not replay_p.is_absolute():
        replay_p = PROJECT_ROOT / replay_p
    if replay_p.is_file():
        token_count = replay_p.stat().st_size // 2
        replay_cache = CompactTokenCache(replay_p, token_count)
        print(f"[REPLAY] Loaded experience replay cache: {replay_p.name} ({token_count:,} tokens)")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    out_p = Path(args.output)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p

    print(f"\n[SFT] Starting Supervised Fine-Tuning on {data_path.name}...")
    t0 = time.time()
    finetune_instructions(
        model=model,
        dataset_path=data_path,
        tokenizer=tokenizer,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        output_checkpoint=str(out_p),
        replay_cache=replay_cache,
        replay_ratio=args.replay_ratio if replay_cache else 0.0,
        max_time=args.max_time,
        save_interval=args.save_interval,
        optimizer_type=args.optimizer,
        muon_lr=args.muon_lr,
        grad_accum_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    print(f"[SFT] Completed in {time.time() - t0:.2f}s. Saved to {out_p.name}")

    # Generate test inference
    print("\n" + "=" * 60)
    print("TESTING CHAT INFERENCE AFTER SFT")
    print("=" * 60)
    gen = TextGenerator(model, tokenizer)
    test_prompts = [
        "What is a database?",
        "What does ACID stand for in database transactions?",
        "What is the difference between SQL and NoSQL databases?",
        "What is the difference between correlation and causation?",
    ]
    s_cfg = SamplingConfig(temperature=0.7, top_k=40, top_p=0.9, max_new_tokens=80)
    for q in test_prompts:
        prompt_formatted = f"### User:\n{q}\n\n### Assistant:\n"
        out = gen.generate(prompt_formatted, config=s_cfg)
        print(f"\nUser: {q}")
        print(f"Assistant: {out.replace(prompt_formatted, '').strip()}")

    if replay_cache:
        replay_cache.close()


if __name__ == "__main__":
    main()

