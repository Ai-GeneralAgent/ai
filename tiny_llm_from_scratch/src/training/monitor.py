from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Any


@dataclass
class MetricRecord:
    step: int
    train_loss: float
    val_loss: Optional[float]
    train_ppl: float
    val_ppl: Optional[float]
    learning_rate: float
    tokens_per_sec: float
    steps_per_sec: float
    elapsed_sec: float
    eta_sec: Optional[float]


class TrainingMonitor:
    """Tracks training metrics, writes JSONL logs, and detects overfitting."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.history: List[MetricRecord] = []
        self.best_val_loss = float("inf")
        self.overfit_counter = 0

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Create or reset log file
            with self.log_path.open("w", encoding="utf-8") as f:
                pass

    def record_step(
        self,
        step: int,
        train_loss: float,
        val_loss: Optional[float],
        lr: float,
        tokens_per_sec: float,
        steps_per_sec: float,
        elapsed_sec: float,
        eta_sec: Optional[float] = None,
    ) -> MetricRecord:
        train_ppl = float("inf") if train_loss > 20 else 2.718281828459045 ** train_loss
        val_ppl = None
        if val_loss is not None:
            val_ppl = float("inf") if val_loss > 20 else 2.718281828459045 ** val_loss

        rec = MetricRecord(
            step=step,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4) if val_loss is not None else None,
            train_ppl=round(train_ppl, 2),
            val_ppl=round(val_ppl, 2) if val_ppl is not None else None,
            learning_rate=lr,
            tokens_per_sec=round(tokens_per_sec, 1),
            steps_per_sec=round(steps_per_sec, 2),
            elapsed_sec=round(elapsed_sec, 1),
            eta_sec=round(eta_sec, 1) if eta_sec is not None else None,
        )
        self.history.append(rec)

        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec)) + "\n")

        # Overfitting check
        if val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.overfit_counter = 0
            else:
                self.overfit_counter += 1

        return rec

    def is_overfitting(self, patience_steps: int = 5) -> bool:
        """Returns True if validation loss has not improved for patience_steps evaluations."""
        return self.overfit_counter >= patience_steps

    def get_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        last = self.history[-1]
        return {
            "total_steps": last.step,
            "final_train_loss": last.train_loss,
            "final_val_loss": last.val_loss,
            "best_val_loss": round(self.best_val_loss, 4) if self.best_val_loss != float("inf") else None,
            "total_elapsed_sec": last.elapsed_sec,
        }

