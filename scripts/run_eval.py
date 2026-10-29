#!/usr/bin/env python3
"""
CLI Script: Run 6-Task Evaluation Benchmark on Any Checkpoint (Section 18)
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tiny_llm_from_scratch"))

import torch
from tokenizers import Tokenizer
from src.models import load_modern_checkpoint
from src.generation import TextGenerator, SamplingConfig
from src.evaluation import BenchmarkSuite
from src.utils import clamp_blas_threads, configure_utf8_console

clamp_blas_threads()
configure_utf8_console()


def main():
    parser = argparse.ArgumentParser(description="Run Standardized Benchmark Suite on Checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/modern_production_v2_best.pt", help="Checkpoint path")
    parser.add_argument("--tokenizer", default="tokenizer_bpe.json", help="Tokenizer path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt_p = PROJECT_ROOT / "tiny_llm_from_scratch" / args.checkpoint
    if not ckpt_p.is_file():
        ckpt_p = Path(args.checkpoint)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    tok_p = PROJECT_ROOT / "tiny_llm_from_scratch" / args.tokenizer
    if not tok_p.is_file():
        tok_p = Path(args.tokenizer)

    tokenizer = Tokenizer.from_file(str(tok_p))
    model, payload = load_modern_checkpoint(ckpt_p, device=args.device)
    generator = TextGenerator(model, tokenizer)

    def generate_fn(prompt: str, max_tokens: int) -> str:
        s_cfg = SamplingConfig(temperature=0.7, top_k=40, top_p=0.9, max_new_tokens=max_tokens)
        return generator.generate(prompt, config=s_cfg, device=args.device)

    print(f"\n[BENCHMARK] Running evaluation suite on: {ckpt_p.name}...")
    res = BenchmarkSuite.run_benchmark(ckpt_p.name, generate_fn)

    print(f"\n========================================================")
    print(f"BENCHMARK RESULTS: {res['model_name']}")
    print(f"Passed: {res['passed_tasks']} / {res['total_tasks']} ({res['score_percentage']}%)")
    print(f"Average Latency: {res['average_latency_sec']}s")
    print(f"========================================================")
    for r in res["results"]:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"  {status} {r['category']}: {r['prompt'][:40]}... -> '{r['output'][:30]}...' ({r['latency_sec']}s)")


if __name__ == "__main__":
    main()
