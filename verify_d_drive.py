import sys
from pathlib import Path
sys.path.insert(0, "tiny_llm_from_scratch")

from src.ui.cli import discover_best_checkpoint, load_model_safely
from src.tools.builtin import solve_math_problem

root = Path("tiny_llm_from_scratch")
best_ckpt = discover_best_checkpoint(root)
print("Active Checkpoint on D: drive:", best_ckpt.name)
model, desc = load_model_safely(best_ckpt, "cpu")
print("Loaded Model:", desc)
print("Config Vocab Size:", model.config.vocab_size)
print("Trainable Parameters:", f"{model.count_parameters():,}")

math_res = solve_math_problem("Solve: 5x + 15 = 45")
print("Math Problem Solution:", math_res["result"])
print(math_res["explanation"])


