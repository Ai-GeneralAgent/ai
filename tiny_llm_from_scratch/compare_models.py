import sys
import time
from pathlib import Path

# Ensure UTF-8 output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    ModelConfig,
    ModernTransformer,
    inspect_checkpoint,
    save_checkpoint,
)
from src.generation import TextGenerator, SamplingConfig
from src.evaluation import BenchmarkSuite
from model_bpe import TinyBPEGPT


def main():
    device = "cpu"
    tok_path = PROJECT_ROOT / "tokenizer_bpe.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    legacy_path = PROJECT_ROOT / "checkpoints" / "tiny_bpe_expanded_continued_v2.pt"
    modern_path = PROJECT_ROOT / "checkpoints" / "modern_tiny_transformer_best.pt"

    print("=" * 65)
    print("[BENCHMARK] EMPIRICAL COMPARISON: LEGACY vs MODERN ARCHITECTURE")
    print("=" * 65)

    # 1. Load Legacy Model
    print(f"\n1. Loading Legacy Model ({legacy_path.name})...")
    ckpt_legacy = torch.load(legacy_path, map_location=device, weights_only=False)
    legacy_model = TinyBPEGPT(
        vocab_size=ckpt_legacy["vocab_size"],
        block_size=ckpt_legacy["block_size"],
        n_embd=ckpt_legacy["n_embd"],
        n_head=ckpt_legacy["n_head"],
        n_layer=ckpt_legacy["n_layer"],
        dropout=ckpt_legacy.get("dropout", 0.1),
    ).to(device)
    legacy_model.load_state_dict(ckpt_legacy["model_state_dict"])
    legacy_model.eval()

    legacy_gen = TextGenerator(legacy_model, tokenizer)
    legacy_val_loss = ckpt_legacy.get("val_loss", 5.766)

    # 2. Setup Modern Model
    print(f"\n2. Setting up Modern Model (v2.0.0 RoPE + SwiGLU + RMSNorm)...")
    if not modern_path.is_file():
        # Train a fast modern checkpoint if not yet trained
        from src.data import CompactTokenCache
        from src.training import Trainer, TrainingConfig

        sample_txt = PROJECT_ROOT / "alice.txt"
        cache = CompactTokenCache.build_or_load(sample_txt, tok_path, tokenizer)
        cfg = ModelConfig.preset_small(vocab_size=tokenizer.get_vocab_size())
        modern_model = ModernTransformer(cfg).to(device)

        t_cfg = TrainingConfig(
            max_steps=50,
            eval_interval=25,
            eval_batches=5,
            batch_size=4,
            output_dir=str(PROJECT_ROOT / "checkpoints"),
            run_name="modern_tiny_transformer",
        )
        trainer = Trainer(modern_model, t_cfg, cache, tok_path)
        trainer.train()
        cache.close()
    else:
        from src.models import load_modern_checkpoint
        modern_model, mod_meta = load_modern_checkpoint(modern_path, device=device)

    modern_gen = TextGenerator(modern_model, tokenizer)

    sampling_cfg = SamplingConfig(temperature=0.7, top_k=40, top_p=0.9)

    def run_legacy_gen(prompt: str, max_tokens: int) -> str:
        s_cfg = SamplingConfig(temperature=0.7, top_k=40, top_p=0.9, max_new_tokens=max_tokens)
        return legacy_gen.generate(prompt, config=s_cfg)

    def run_modern_gen(prompt: str, max_tokens: int) -> str:
        s_cfg = SamplingConfig(temperature=0.7, top_k=40, top_p=0.9, max_new_tokens=max_tokens)
        return modern_gen.generate(prompt, config=s_cfg)

    # Run Benchmark on Legacy
    print("\nRunning Benchmark Suite on Legacy Model...")
    res_legacy = BenchmarkSuite.run_benchmark("Legacy TinyBPEGPT (v1)", run_legacy_gen)

    # Run Benchmark on Modern
    print("Running Benchmark Suite on Modern Model...")
    res_modern = BenchmarkSuite.run_benchmark("Modern Transformer (v2)", run_modern_gen)

    # Print Side-by-Side Report
    print("\n" + "=" * 65)
    print("EMPIRICAL BENCHMARK REPORT (HONEST & UNBIASED)")
    print("=" * 65)
    print(f"{'Metric':<28} | {'Legacy Model (v1)':<16} | {'Modern Model (v2)':<16}")
    print("-" * 65)
    print(f"{'Architecture':<28} | {'AbsPos+GELU+LN':<16} | {'RoPE+SwiGLU+RMS':<16}")
    mod_params_str = f"{modern_model.count_parameters():,}"
    leg_score_str = f"{res_legacy['score_percentage']}%"
    mod_score_str = f"{res_modern['score_percentage']}%"
    leg_lat_str = f"{res_legacy['average_latency_sec']}s"
    mod_lat_str = f"{res_modern['average_latency_sec']}s"
    leg_loss_str = f"{legacy_val_loss:.4f}"

    print(f"{'Parameters':<28} | {'2,924,928':<16} | {mod_params_str:<16}")
    print(f"{'Best Val Loss':<28} | {leg_loss_str:<16} | {'Recorded in run':<16}")
    print(f"{'Benchmark Score':<28} | {leg_score_str:<16} | {mod_score_str:<16}")
    print(f"{'Avg Latency (per prompt)':<28} | {leg_lat_str:<16} | {mod_lat_str:<16}")
    print("-" * 65)

    print("\nDetailed Task Comparison:")
    for l_task, m_task in zip(res_legacy["results"], res_modern["results"]):
        print(f"\n[{l_task['category']}]")
        print(f"  Prompt:   {l_task['prompt']}")
        print(f"  Expected: {l_task['expected']}")
        print(f"  Legacy:   {repr(l_task['output'][:60])} (Passed: {l_task['passed']})")
        print(f"  Modern:   {repr(m_task['output'][:60])} (Passed: {m_task['passed']})")


if __name__ == "__main__":
    main()
