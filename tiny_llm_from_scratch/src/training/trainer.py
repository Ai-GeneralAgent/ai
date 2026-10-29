from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from ..models.config import ModelConfig
from ..models.transformer import ModernTransformer
from ..models.checkpoint import save_checkpoint, load_modern_checkpoint, compute_file_fingerprint
from ..data.token_cache import CompactTokenCache
from .monitor import TrainingMonitor


@dataclass
class TrainingConfig:
    max_steps: int = 1000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    batch_size: int = 4
    grad_accum_steps: int = 1
    eval_interval: int = 100
    eval_batches: int = 10
    save_interval: int = 500
    weight_decay: float = 0.1
    clip_grad_norm: float = 1.0
    patience: int = 10
    seed: int = 1337
    device: str = "cpu"
    output_dir: str = "checkpoints"
    run_name: str = "modern_tiny_transformer"


class CosineWarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float,
        min_lr: float,
        start_step: int = 0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.start_step = start_step

    def step(self, current_step: int) -> float:
        eff_step = max(0, current_step - self.start_step)
        total_steps = max(1, self.max_steps - self.start_step)
        warmup = min(self.warmup_steps, max(1, total_steps // 4))

        if eff_step < warmup:
            lr = self.min_lr + (self.max_lr - self.min_lr) * (eff_step + 1) / max(1, warmup)
        elif current_step >= self.max_steps:
            lr = self.min_lr
        else:
            decay_ratio = (eff_step - warmup) / max(1, total_steps - warmup)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class Trainer:
    """Production-grade Transformer trainer with atomic checkpoints, recovery, and monitor."""

    def __init__(
        self,
        model: ModernTransformer,
        train_config: TrainingConfig,
        token_cache: CompactTokenCache,
        tokenizer_path: Union[str, Path],
        val_cache: Optional[CompactTokenCache] = None,
        val_split_ratio: float = 0.1,
    ):
        self.model = model.to(train_config.device)
        self.config = train_config
        self.cache = token_cache
        self.tokenizer_path = Path(tokenizer_path)

        torch.manual_seed(train_config.seed)

        # Setup dataset splits
        block_size = self.model.config.block_size
        if val_cache is not None:
            self.train_start, self.train_end = 0, token_cache.token_count
            self.val_cache = val_cache
            self.val_start, self.val_end = 0, val_cache.token_count
        else:
            split_idx = int(token_cache.token_count * (1.0 - val_split_ratio))
            self.train_start = 0
            self.train_end = split_idx
            self.val_cache = token_cache
            self.val_end = token_cache.token_count
            # Ensure validation split has at least (block_size + 2) tokens
            if (self.val_end - split_idx) <= block_size + 1:
                self.val_start = max(0, self.val_end - (block_size + 2))
                self.train_end = self.val_start if self.val_start > block_size + 1 else self.val_end
            else:
                self.val_start = split_idx

        # Optimizer: AdamW with weight decay on 2D parameters (weights) only
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            max_lr=train_config.learning_rate,
            min_lr=train_config.min_learning_rate,
        )

        out_path = Path(train_config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.log_path = out_path / f"{train_config.run_name}_metrics.jsonl"
        self.best_ckpt_path = out_path / f"{train_config.run_name}_best.pt"
        self.latest_ckpt_path = out_path / f"{train_config.run_name}_latest.pt"

        self.monitor = TrainingMonitor(self.log_path)
        self.global_step = 0
        self.best_val_loss = float("inf")

    def set_resume_state(self, step: int, best_val_loss: float):
        self.global_step = step
        self.best_val_loss = best_val_loss
        self.monitor.best_val_loss = best_val_loss
        self.scheduler.start_step = step

    def print_preflight(self):
        cfg = self.model.config
        total_tokens = self.train_end - self.train_start
        eff_batch = self.config.batch_size * self.config.grad_accum_steps * cfg.block_size

        print("\n" + "=" * 60)
        print("[CONFIG] PRE-FLIGHT TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Device:               {self.config.device}")
        print(f"Model Architecture:   {cfg.arch_version} (Layers: {cfg.n_layer}, Embd: {cfg.n_embd}, Heads: {cfg.n_head})")
        print(f"Norm / Pos / FFN:     {cfg.norm_type} / {cfg.pos_emb} / {cfg.ffn_type}")
        print(f"Trainable Parameters: {self.model.count_parameters():,}")
        print(f"Context Length:       {cfg.block_size} tokens")
        print(f"Training Tokens:      {total_tokens:,}")
        print(f"Batch Size:           {self.config.batch_size} (Grad Accum: {self.config.grad_accum_steps})")
        print(f"Effective Batch:      {eff_batch:,} tokens/step")
        print(f"Max Planned Steps:    {self.config.max_steps:,}")
        print(f"Learning Rate Range:  {self.config.learning_rate} -> {self.config.min_learning_rate}")
        print(f"Checkpoints:          {self.best_ckpt_path.resolve()}")
        print("=" * 60 + "\n")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        batches = self.config.eval_batches
        block_size = self.model.config.block_size

        for _ in range(batches):
            x, y = self.val_cache.sample_batch(
                self.val_start, self.val_end, self.config.batch_size, block_size, self.config.device
            )
            _, loss = self.model(x, y)
            total_loss += loss.item()

        self.model.train()
        return total_loss / max(1, batches)

    def train(self) -> TrainingMonitor:
        self.print_preflight()
        self.model.train()
        block_size = self.model.config.block_size

        started_at = time.time()
        step_timer = time.time()
        tokens_seen = 0

        print(f"Training starting at step {self.global_step + 1}...")
        rolling_loss: Optional[float] = None

        try:
            for step in range(self.global_step + 1, self.config.max_steps + 1):
                self.global_step = step
                lr = self.scheduler.step(step)

                # Gradient accumulation
                self.optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                skip_step = False

                for _ in range(self.config.grad_accum_steps):
                    x, y = self.cache.sample_batch(
                        self.train_start, self.train_end, self.config.batch_size, block_size, self.config.device
                    )
                    _, loss = self.model(x, y)
                    loss_item = loss.item()

                    # Numerical instability / NaN guard (Problem #1 & #6)
                    if math.isnan(loss_item) or math.isinf(loss_item):
                        print(f"  [SPIKE GUARD] Numerical NaN/Inf encountered at step {step}. Skipping step.")
                        skip_step = True
                        break

                    loss = loss / self.config.grad_accum_steps
                    loss.backward()
                    accum_loss += loss.item()
                    tokens_seen += x.numel()

                if skip_step:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Loss Spike Detection (Problem #1)
                if rolling_loss is not None and accum_loss > 3.0 * rolling_loss and step > 10:
                    print(f"  [LOSS SPIKE GUARD] Sudden loss explosion ({accum_loss:.4f} vs rolling {rolling_loss:.4f}) at step {step}. Clamping update.")
                    # Clamp gradients more aggressively
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                elif self.config.clip_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        print(f"  [GRAD SPIKE GUARD] NaN gradient norm at step {step}. Skipping step.")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                self.optimizer.step()
                rolling_loss = accum_loss if rolling_loss is None else (0.95 * rolling_loss + 0.05 * accum_loss)

                # Step metrics
                step_elapsed = max(time.time() - step_timer, 0.001)
                steps_per_sec = 1.0 / step_elapsed
                tokens_per_sec = (self.config.batch_size * self.config.grad_accum_steps * block_size) / step_elapsed
                step_timer = time.time()

                # Periodic Evaluation
                is_eval_step = (step % self.config.eval_interval == 0) or (step == self.config.max_steps)
                val_loss = None

                if is_eval_step:
                    val_loss = self.evaluate()
                    elapsed_total = time.time() - started_at
                    eta_sec = (self.config.max_steps - step) / max(steps_per_sec, 0.01)

                    rec = self.monitor.record_step(
                        step=step,
                        train_loss=accum_loss,
                        val_loss=val_loss,
                        lr=lr,
                        tokens_per_sec=tokens_per_sec,
                        steps_per_sec=steps_per_sec,
                        elapsed_sec=elapsed_total,
                        eta_sec=eta_sec,
                    )

                    print(
                        f"Step {step:5d}/{self.config.max_steps} | "
                        f"Train Loss: {accum_loss:.4f} (ppl: {rec.train_ppl:.1f}) | "
                        f"Val Loss: {val_loss:.4f} (ppl: {rec.val_ppl:.1f}) | "
                        f"LR: {lr:.2e} | "
                        f"{tokens_per_sec:.0f} tok/s | ETA: {int(eta_sec)}s"
                    )

                    # Save Best Checkpoint
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            self.model,
                            self.best_ckpt_path,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            step=step,
                            val_loss=val_loss,
                            tokenizer_path=str(self.tokenizer_path),
                        )
                        print(f"  [CHECKPOINT] New best checkpoint saved: {self.best_ckpt_path.name} (val_loss: {val_loss:.4f})")

                    # Overfitting detection
                    if self.monitor.is_overfitting(self.config.patience):
                        print(f"\n[WARNING] Early stopping triggered at step {step}: val_loss has not improved for {self.config.patience} evaluations.")
                        break

                # Periodic latest checkpoint
                if step % self.config.save_interval == 0 or step == self.config.max_steps:
                    save_checkpoint(
                        self.model,
                        self.latest_ckpt_path,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=step,
                        val_loss=val_loss,
                        tokenizer_path=str(self.tokenizer_path),
                    )

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Training interrupted by user. Saving emergency checkpoint...")
            save_checkpoint(
                self.model,
                self.latest_ckpt_path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=self.global_step,
                tokenizer_path=str(self.tokenizer_path),
            )
            print(f"Saved state to: {self.latest_ckpt_path}")

        print(f"\nTraining finished in {time.time() - started_at:.1f}s. Best val loss: {self.best_val_loss:.4f}")
        return self.monitor
