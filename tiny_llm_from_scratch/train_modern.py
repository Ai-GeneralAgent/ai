import argparse
from pathlib import Path
import sys
import time

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ModelConfig, ModernTransformer, save_checkpoint
from src.data import CompactTokenCache
from src.training import Trainer, TrainingConfig
from src.generation import TextGenerator, SamplingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Modern Transformer (v2.0.0)")
    parser.add_argument("--data", default="data_knowledge_corpus.txt", help="Training corpus file or .u16 cache")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--run-name", default="modern_production_v2")
    parser.add_argument("--device", default="auto", help="Compute device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    # Device auto-detection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[DEVICE] Compute device: {device.upper()}")

    tok_path = PROJECT_ROOT / "tokenizer_bpe.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Resolve dataset or cache file with robust multi-device fallback
    data_arg = Path(args.data)
    if not data_arg.is_absolute():
        data_arg = PROJECT_ROOT / data_arg

    cache = None
    if data_arg.suffix == ".u16" and data_arg.is_file():
        token_count = data_arg.stat().st_size // 2
        cache = CompactTokenCache(data_arg, token_count)
        print(f"[DATA] Loaded direct token cache: {data_arg.name} ({token_count:,} tokens)")
    else:
        target_file = data_arg
        if not target_file.is_file():
            potential_u16 = target_file.with_suffix(".bpe_tokens.u16")
            if potential_u16.is_file():
                token_count = potential_u16.stat().st_size // 2
                cache = CompactTokenCache(potential_u16, token_count)
                print(f"[DATA] Found token cache: {potential_u16.name} ({token_count:,} tokens)")
            else:
                fallback_file = PROJECT_ROOT / "alice.txt"
                if fallback_file.is_file():
                    print(f"[DATA] '{data_arg.name}' not found. Falling back to default corpus: {fallback_file.name}")
                    target_file = fallback_file
                else:
                    raise FileNotFoundError(f"Neither '{data_arg}' nor '{fallback_file}' could be found.")

        if cache is None:
            print(f"[DATA] Preparing token cache for: {target_file.name}...")
            cache = CompactTokenCache.build_or_load(target_file, tok_path, tokenizer=tokenizer)
            print(f"[DATA] Ready. Total tokens: {cache.token_count:,}")

    initial_step = 0
    best_val_loss = float("inf")
    if args.resume:
        resume_p = Path(args.resume)
        if not resume_p.is_absolute():
            resume_p = PROJECT_ROOT / resume_p
        from src.models import load_modern_checkpoint
        model, payload = load_modern_checkpoint(resume_p, device=device)
        initial_step = payload.get("step", 0)
        best_val_loss = payload.get("val_loss", float("inf"))
        print(f"[RESUME] Loaded model from: {resume_p.name} (step {initial_step}, val_loss {best_val_loss})")
    else:
        cfg = ModelConfig.preset_small(vocab_size=tokenizer.get_vocab_size())
        cfg.block_size = args.block_size
        model = ModernTransformer(cfg)

    eval_int = args.eval_interval if args.eval_interval != 25 else (200 if args.steps >= 2000 else 25)
    t_cfg = TrainingConfig(
        max_steps=args.steps,
        learning_rate=args.lr,
        min_learning_rate=args.lr * 0.1,
        warmup_steps=min(500, max(10, args.steps // 10)),
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        eval_interval=eval_int,
        eval_batches=10,
        save_interval=min(500, args.steps),
        output_dir=str(PROJECT_ROOT / "checkpoints"),
        run_name=args.run_name,
        device=device,
    )

    trainer = Trainer(model, t_cfg, cache, tok_path)
    if initial_step > 0:
        trainer.global_step = initial_step
        trainer.best_val_loss = best_val_loss
        trainer.monitor.best_val_loss = best_val_loss

    monitor = trainer.train()

    # Generate sample output from trained model
    print("\n" + "=" * 60)
    print("GENERATING SAMPLE FROM TRAINED MODERN MODEL")
    print("=" * 60)
    gen = TextGenerator(model, tokenizer)
    prompts = [
        "Artificial intelligence is",
        "The universe is composed of",
    ]
    s_cfg = SamplingConfig(temperature=0.75, top_k=40, top_p=0.9, max_new_tokens=60)
    for p in prompts:
        out = gen.generate(p, config=s_cfg)
        print(f"\nPrompt: '{p}'")
        print(f"Output: {out.strip()}")

    cache.close()
    print("\nTraining section successfully completed!")


if __name__ == "__main__":
    main()

