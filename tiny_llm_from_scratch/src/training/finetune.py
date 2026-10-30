import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from ..models.transformer import ModernTransformer
from ..models.checkpoint import save_checkpoint
from ..data.token_cache import CompactTokenCache


class InstructionDataset:
    """
    Industrial SFT Dataset with:
    - Multi-turn & single-turn ChatML / OpenAI message format parsing
    - Completion-only loss masking (-100 for user prompts and system context)
    - Experience Replay support to prevent catastrophic forgetting
    """

    TEMPLATE = "### User:\n{instruction}\n\n### Assistant:\n{response}"

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: Tokenizer,
        block_size: int = 256,
        replay_cache: Optional[CompactTokenCache] = None,
        replay_ratio: float = 0.20,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.replay_cache = replay_cache
        self.replay_ratio = replay_ratio
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"Instruction dataset not found: {path.resolve()}")

        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    # 1. Check for standard OpenAI/Anthropic "messages" format
                    if "messages" in obj and isinstance(obj["messages"], list):
                        msgs = obj["messages"]
                        system_prompt = ""
                        user_prompt = ""
                        assistant_response = ""
                        for m in msgs:
                            role = m.get("role", "")
                            content = m.get("content", "")
                            if role == "system":
                                system_prompt = content
                            elif role == "user":
                                user_prompt = content
                            elif role == "assistant":
                                assistant_response = content
                        if user_prompt and assistant_response:
                            self._process_pair(user_prompt, assistant_response, system_prompt=system_prompt)
                            continue

                    # 2. Check for standard instruction / response format
                    inst = obj.get("instruction") or obj.get("prompt") or obj.get("question")
                    resp = obj.get("response") or obj.get("output") or obj.get("answer") or obj.get("completion")
                    if inst and resp:
                        self._process_pair(str(inst), str(resp))
                except Exception:
                    continue

    def _process_pair(self, instruction: str, response: str, system_prompt: str = ""):
        if not system_prompt:
            system_prompt = "You are LLM, a helpful, friendly, and honest assistant built by Yuvraj Thakur."
        prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt_str = f"{prefix}### User:\n{instruction}\n\n### Assistant:\n"
        full_str = f"{prompt_str}{response}\n"

        prompt_ids = self.tokenizer.encode(prompt_str).ids
        full_ids = self.tokenizer.encode(full_str).ids

        if len(full_ids) > self.block_size + 1:
            full_ids = full_ids[: self.block_size + 1]

        x_ids = full_ids[:-1]
        y_ids = full_ids[1:]

        # Create target tensor with -100 for prompt tokens (Prompt Loss Masking)
        prompt_len = min(len(prompt_ids) - 1, len(y_ids))
        targets = list(y_ids)
        for i in range(prompt_len):
            targets[i] = -100

        x_tensor = torch.tensor(x_ids, dtype=torch.long)
        y_tensor = torch.tensor(targets, dtype=torch.long)
        self.examples.append((x_tensor, y_tensor))

    def __len__(self) -> int:
        return len(self.examples)

    def sample_batch(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples a batch with optional base corpus experience replay to prevent forgetting."""
        # Check if experience replay from base corpus is active
        replay_count = 0
        if self.replay_cache and self.replay_ratio > 0:
            replay_count = max(1, int(batch_size * self.replay_ratio))

        sft_count = batch_size - replay_count
        batch_x = []
        batch_y = []

        if sft_count > 0 and len(self.examples) > 0:
            indices = torch.randint(0, len(self.examples), (sft_count,))
            max_len = max(self.examples[i][0].size(0) for i in indices)

            for i in indices:
                x, y = self.examples[i]
                pad_len = max_len - x.size(0)
                if pad_len > 0:
                    x = F.pad(x, (0, pad_len), value=0)
                    y = F.pad(y, (0, pad_len), value=-100)
                batch_x.append(x)
                batch_y.append(y)

        # Experience replay rows from pretraining corpus (normal next-token prediction)
        if replay_count > 0 and self.replay_cache:
            rx, ry = self.replay_cache.sample_batch(
                0, self.replay_cache.token_count, replay_count, self.block_size, device="cpu"
            )
            for i in range(replay_count):
                batch_x.append(rx[i])
                batch_y.append(ry[i])

        # Align lengths if needed
        max_b_len = max(t.size(0) for t in batch_x)
        final_x = []
        final_y = []
        for x, y in zip(batch_x, batch_y):
            pad = max_b_len - x.size(0)
            if pad > 0:
                final_x.append(F.pad(x, (0, pad), value=0))
                final_y.append(F.pad(y, (0, pad), value=-100))
            else:
                final_x.append(x)
                final_y.append(y)

        return torch.stack(final_x).to(device), torch.stack(final_y).to(device)


def finetune_instructions(
    model: ModernTransformer,
    dataset_path: Union[str, Path],
    tokenizer: Tokenizer,
    steps: int = 200,
    lr: float = 1e-4,
    batch_size: int = 4,
    device: str = "cpu",
    output_checkpoint: str = "checkpoints/modern_chat_finetuned.pt",
    replay_cache: Optional[CompactTokenCache] = None,
    replay_ratio: float = 0.20,
    max_time: Optional[float] = None,
    save_interval: int = 250,
    optimizer_type: str = "adamw",
    muon_lr: float = 0.01,
) -> ModernTransformer:
    """
    Industrial Supervised Fine-Tuning (SFT) Loop with:
    - Completion-only loss masking (-100 for prompt tokens)
    - Anti-catastrophic forgetting via replay buffer blending
    - Muon / AdamW hybrid optimization option
    - Global gradient norm clipping (1.0)
    - Loss spike and NaN/Inf guards
    - Wall-clock training duration limit (max_time in seconds)
    - Periodic and best-loss checkpoint saving
    """
    model.train()
    dataset = InstructionDataset(
        dataset_path,
        tokenizer,
        block_size=model.config.block_size,
        replay_cache=replay_cache,
        replay_ratio=replay_ratio,
    )
    if len(dataset) == 0:
        raise ValueError(f"No valid instruction pairs found in {dataset_path}")

    if optimizer_type == "muon":
        from .muon import HybridMuonAdamW
        muon_params = []
        adam_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and "tok_embeddings" not in name and "output" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)

        decay_adam = [p for p in adam_params if p.dim() >= 2]
        nodecay_adam = [p for p in adam_params if p.dim() < 2]
        adam_groups = [
            {"params": decay_adam, "weight_decay": 0.01},
            {"params": nodecay_adam, "weight_decay": 0.0},
        ]
        optimizer = HybridMuonAdamW(muon_params, adam_groups, lr=lr, muon_lr=muon_lr)
        print(f"[OPTIMIZER] Hybrid Muon + AdamW active (Muon LR: {muon_lr}, AdamW LR: {lr}).")
    else:
        # Decouple weight decay: 2D weights get decay, 1D layer norms & biases get 0.0
        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        print(f"[OPTIMIZER] Standard AdamW active (LR: {lr}).")

    time_str = f", Max Duration: {max_time/3600:.1f} hours" if max_time else ""
    print(f"Finetuning model on {len(dataset):,} instruction pairs for {steps:,} steps (Replay ratio: {replay_ratio*100:.0f}%{time_str})...")

    rolling_loss = None
    best_loss = float("inf")
    start_time = time.time()
    last_step = 0

    for step in range(1, steps + 1):
        elapsed = time.time() - start_time
        if max_time and elapsed >= max_time:
            print(f"\n[TIME LIMIT REACHED] Trained for {elapsed/3600:.2f} hours ({elapsed:.0f}s). Stopping gracefully at step {step-1}.")
            last_step = step - 1
            break

        x, y = dataset.sample_batch(batch_size, device=device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

        # Numerical stability & Loss spike protection
        loss_val = loss.item()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARNING] NaN/Inf loss encountered at step {step}. Skipping step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if rolling_loss is not None and loss_val > 3.0 * rolling_loss:
            print(f"  [SPIKE GUARD] Loss spike ({loss_val:.4f} vs rolling {rolling_loss:.4f}) at step {step}. Skipping step.")
            optimizer.zero_grad(set_to_none=True)
            continue

        rolling_loss = loss_val if rolling_loss is None else 0.9 * rolling_loss + 0.1 * loss_val

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_step = step

        # Periodic status logging
        if step % 50 == 0 or step == steps:
            cur_elapsed = time.time() - start_time
            rate = step / max(1.0, cur_elapsed)
            eta_str = ""
            if max_time:
                remaining_time = max(0.0, max_time - cur_elapsed)
                eta_str = f" | Remaining: {remaining_time/60:.1f}m"
            print(f"Finetune Step {step:5d}/{steps} | Loss: {loss.item():.4f} (Rolling: {rolling_loss:.4f}) | {rate:.2f} steps/s | Elapsed: {cur_elapsed/60:.1f}m{eta_str}", flush=True)

        # Save checkpoint periodically
        if step % save_interval == 0:
            save_checkpoint(model, output_checkpoint, optimizer=optimizer, step=step, val_loss=rolling_loss)
            is_new_best = False
            if rolling_loss < best_loss:
                best_loss = rolling_loss
                best_ckpt = str(Path(output_checkpoint).with_name(Path(output_checkpoint).stem + "_best.pt"))
                save_checkpoint(model, best_ckpt, optimizer=optimizer, step=step, val_loss=best_loss)
                is_new_best = True
            best_tag = " (NEW BEST!)" if is_new_best else ""
            print(f"  [CHECKPOINT] Step {step} saved to {Path(output_checkpoint).name}{best_tag} | Loss: {rolling_loss:.4f}", flush=True)

    final_loss = rolling_loss if rolling_loss is not None else 0.0
    save_checkpoint(model, output_checkpoint, optimizer=optimizer, step=last_step, val_loss=final_loss)
    print(f"Finetuned checkpoint saved to {output_checkpoint}")
    return model

