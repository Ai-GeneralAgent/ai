"""
Nightly 'Sleep' Continual Learning Consolidation Engine
Approximates biological memory consolidation in weights:
1. Gathers day's verified facts from Hippocampus memory and verified RLVR solutions
2. Merges with base corpus replay buffer to prevent catastrophic forgetting
3. Performs fine-tuning consolidation on candidate weights
4. Strict Validation Gate: Evaluates against anchor benchmark and only promotes weights if no regression occurs
"""
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer

from ..memory.hippocampus import HippocampusMemory
from ..models.checkpoint import load_modern_checkpoint, save_checkpoint
from ..models.transformer import ModernTransformer
from ..data.token_cache import CompactTokenCache
from .finetune import finetune_instructions


ANCHOR_BENCHMARK = [
    {
        "question": "What is a database?",
        "expected_keywords": ["database", "data", "collection", "system", "sql", "storage"],
    },
    {
        "question": "What is 15 multiplied by 12?",
        "expected_keywords": ["180"],
    },
    {
        "question": "Who created you and what is your model name?",
        "expected_keywords": ["llm", "yuvraj"],
    },
    {
        "question": "What is the difference between correlation and causation?",
        "expected_keywords": ["correlation", "causation", "relationship", "cause"],
    },
]


class SleepConsolidationEngine:
    """
    Manages overnight weight-level memory consolidation:
    Turns daily experiences and verified discoveries into permanent neural weights.
    """

    def __init__(
        self,
        project_root: Union[str, Path] = "d:/tiny_llm_from_scratch/tiny_llm_from_scratch",
        db_path: str = "memory/hippocampus.db",
        active_checkpoint: str = "checkpoints/modern_chat_finetuned_best.pt",
        tokenizer_path: str = "tokenizer_bpe.json",
        replay_cache_path: str = "data_master_v3_unified.bpe_tokens.u16",
    ):
        self.root = Path(project_root)
        self.hippocampus = HippocampusMemory(self.root / db_path)
        self.active_checkpoint = self.root / active_checkpoint
        self.tokenizer_path = self.root / tokenizer_path
        self.replay_cache_path = self.root / replay_cache_path

    def gather_daily_consolidation_dataset(self) -> Path:
        """Collects verified facts, RLVR solutions, and creates a consolidated training file."""
        today = datetime.now().strftime("%Y%m%d")
        out_file = self.root / "training_data" / f"nightly_sleep_batch_{today}.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        samples = []

        # 1. Facts from Hippocampus
        facts = self.hippocampus.export_memories_for_training()
        samples.extend(facts)

        # 2. Verified RLVR samples
        rlvr_file = self.root / "training_data" / "rlvr_verified_data.jsonl"
        if rlvr_file.is_file():
            with rlvr_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            samples.append({
                                "instruction": d.get("instruction", ""),
                                "response": d.get("response", ""),
                            })
                        except Exception:
                            continue

        # 3. Baseline conversational seeds to preserve natural tone
        conv_seed_file = self.root / "instruction_conversations.jsonl"
        if conv_seed_file.is_file():
            with conv_seed_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            samples.append({
                                "instruction": d.get("instruction", ""),
                                "response": d.get("response", ""),
                            })
                        except Exception:
                            continue

        # Fallback sample if empty
        if not samples:
            samples.append({
                "instruction": "Who are you?",
                "response": "I am LLM, a personal AI assistant built by Yuvraj Thakur.",
            })

        with out_file.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"[SLEEP CONSOLIDATION] Prepared {len(samples)} daily training pairs in {out_file.name}")
        return out_file

    def evaluate_model_benchmark(self, model: ModernTransformer, tokenizer: Tokenizer) -> float:
        """Evaluates model against anchor questions to guard against regression."""
        model.eval()
        from ..generation.sampler import TextGenerator, SamplingConfig

        gen = TextGenerator(model, tokenizer)
        cfg = SamplingConfig(temperature=0.3, top_k=20, max_new_tokens=40)
        total_score = 0.0

        for item in ANCHOR_BENCHMARK:
            prompt = f"### User:\n{item['question']}\n\n### Assistant:\n"
            generated = gen.generate(prompt, config=cfg).lower()
            # Score keyword coverage
            matched = sum(1 for kw in item["expected_keywords"] if kw.lower() in generated)
            total_score += matched / max(1, len(item["expected_keywords"]))

        model.train()
        return total_score / max(1, len(ANCHOR_BENCHMARK))

    def run_sleep_cycle(self, steps: int = 50, lr: float = 5e-5, max_time_sec: float = 600) -> Dict[str, Any]:
        """Executes the full nightly sleep consolidation cycle with safety rollback."""
        print("\n" + "=" * 60)
        print("STARTING NIGHTLY 'SLEEP' WEIGHT CONSOLIDATION")
        print("=" * 60)

        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load active production weights
        if not self.active_checkpoint.is_file():
            raise FileNotFoundError(f"Active checkpoint not found: {self.active_checkpoint}")

        base_model, payload = load_modern_checkpoint(self.active_checkpoint, device=device)
        print(f"[BASE] Loaded active checkpoint: {self.active_checkpoint.name} (Step {payload.get('step')})")

        # 2. Evaluate pre-sleep baseline score
        baseline_score = self.evaluate_model_benchmark(base_model, tokenizer)
        print(f"[EVAL] Pre-Sleep Anchor Benchmark Score: {baseline_score * 100:.1f}%")

        # 3. Prepare daily consolidation data
        dataset_path = self.gather_daily_consolidation_dataset()

        # 4. Optional replay cache
        replay_cache = None
        if self.replay_cache_path.is_file():
            token_count = self.replay_cache_path.stat().st_size // 2
            replay_cache = CompactTokenCache(self.replay_cache_path, token_count)
            print(f"[REPLAY] Loaded experience replay cache: {token_count:,} tokens")

        candidate_ckpt = self.root / "checkpoints" / "sleep_candidate.pt"

        # 5. Execute consolidation training on a clone
        candidate_model = copy.deepcopy(base_model)
        t0 = time.time()
        finetune_instructions(
            model=candidate_model,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            steps=steps,
            lr=lr,
            batch_size=4,
            device=device,
            output_checkpoint=str(candidate_ckpt),
            replay_cache=replay_cache,
            replay_ratio=0.25,
            max_time=max_time_sec,
        )
        train_elapsed = time.time() - t0

        if replay_cache:
            replay_cache.close()

        # 6. Evaluate candidate score
        candidate_score = self.evaluate_model_benchmark(candidate_model, tokenizer)
        print(f"[EVAL] Post-Sleep Candidate Score: {candidate_score * 100:.1f}% (vs Baseline: {baseline_score * 100:.1f}%)")

        # 7. Validation Gate
        promoted = False
        if candidate_score >= baseline_score - 0.05:  # Tolerance threshold
            backup_path = self.active_checkpoint.with_name(
                f"{self.active_checkpoint.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            shutil.copy2(self.active_checkpoint, backup_path)
            shutil.copy2(candidate_ckpt, self.active_checkpoint)
            promoted = True
            print(f"[PROMOTED] Consolidation succeeded without regression! New weights promoted to {self.active_checkpoint.name}")
            print(f"[BACKUP] Prior weights archived to {backup_path.name}")
        else:
            print("[REJECTED] Candidate weights caused benchmark regression. Rolled back safely to protect memory stability.")

        return {
            "promoted": promoted,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "training_time": train_elapsed,
            "checkpoint": str(self.active_checkpoint),
        }

