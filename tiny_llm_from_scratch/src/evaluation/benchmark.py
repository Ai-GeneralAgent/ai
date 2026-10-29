import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional
import torch


@dataclass
class BenchmarkTask:
    category: str
    prompt: str
    expected_pattern: str  # regex or keyword that must appear
    max_tokens: int = 50


class BenchmarkSuite:
    """Standardized versioned benchmark suite for evaluating local Transformers."""

    TASKS: List[BenchmarkTask] = [
        BenchmarkTask(
            category="English Fluency",
            prompt="The quick brown fox jumps over the lazy",
            expected_pattern="dog",
            max_tokens=20,
        ),
        BenchmarkTask(
            category="Instruction Following",
            prompt="### Instruction:\nAnswer in one word. What is the capital of France?\n\n### Response:\n",
            expected_pattern="paris",
            max_tokens=15,
        ),
        BenchmarkTask(
            category="Mathematics",
            prompt="What is 15 * 3?",
            expected_pattern="45",
            max_tokens=25,
        ),
        BenchmarkTask(
            category="Code Generation",
            prompt="Write a Python function named add that returns x + y.",
            expected_pattern="def add",
            max_tokens=40,
        ),
        BenchmarkTask(
            category="Summarization",
            prompt="Summarize in short: Artificial Intelligence is a branch of computer science focused on building smart machines capable of performing tasks that typically require human intelligence.",
            expected_pattern="intelligence",
            max_tokens=30,
        ),
        BenchmarkTask(
            category="Tool Selection",
            prompt="Calculate the square root of 144.",
            expected_pattern="12",
            max_tokens=25,
        ),
    ]

    @classmethod
    def run_benchmark(
        cls,
        model_name: str,
        generate_fn: Callable[[str, int], str],
    ) -> Dict[str, Any]:
        """Runs the standard benchmark suite on a model generator function."""
        results = []
        passed_count = 0
        total_time = 0.0

        for task in cls.TASKS:
            t0 = time.time()
            output = generate_fn(task.prompt, task.max_tokens)
            elapsed = time.time() - t0
            total_time += elapsed

            passed = task.expected_pattern.lower() in output.lower()
            if passed:
                passed_count += 1

            results.append({
                "category": task.category,
                "prompt": task.prompt,
                "output": output.strip(),
                "expected": task.expected_pattern,
                "passed": passed,
                "latency_sec": round(elapsed, 3),
            })

        avg_latency = total_time / len(cls.TASKS) if cls.TASKS else 0.0
        score_pct = round(100.0 * passed_count / len(cls.TASKS), 1)

        return {
            "model_name": model_name,
            "total_tasks": len(cls.TASKS),
            "passed_tasks": passed_count,
            "score_percentage": score_pct,
            "average_latency_sec": round(avg_latency, 3),
            "results": results,
        }

