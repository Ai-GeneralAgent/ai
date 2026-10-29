import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from ..models.transformer import ModernTransformer
from ..models.checkpoint import save_checkpoint


class InstructionDataset:
    """Prepares prompt-response pairs with prompt loss masking (-100)."""

    TEMPLATE = "### User:\n{instruction}\n\n### Assistant:\n{response}"

    def __init__(self, jsonl_path: Union[str, Path], tokenizer: Tokenizer, block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
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
                    inst = obj.get("instruction") or obj.get("prompt")
                    resp = obj.get("response") or obj.get("output") or obj.get("answer")
                    if inst and resp:
                        self._process_pair(inst, resp)
                except Exception:
                    continue

    def _process_pair(self, instruction: str, response: str):
        prompt_str = f"### User:\n{instruction}\n\n### Assistant:\n"
        full_str = f"{prompt_str}{response}\n"

        prompt_ids = self.tokenizer.encode(prompt_str).ids
        full_ids = self.tokenizer.encode(full_str).ids

        if len(full_ids) > self.block_size + 1:
            full_ids = full_ids[: self.block_size + 1]

        x_ids = full_ids[:-1]
        y_ids = full_ids[1:]

        # Create target tensor with -100 for prompt tokens
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
        indices = torch.randint(0, len(self.examples), (batch_size,))
        batch_x = []
        batch_y = []
        max_len = max(self.examples[i][0].size(0) for i in indices)

        for i in indices:
            x, y = self.examples[i]
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                x = F.pad(x, (0, pad_len), value=0)
                y = F.pad(y, (0, pad_len), value=-100)
            batch_x.append(x)
            batch_y.append(y)

        return torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)


def finetune_instructions(
    model: ModernTransformer,
    dataset_path: Union[str, Path],
    tokenizer: Tokenizer,
    steps: int = 200,
    lr: float = 1e-4,
    batch_size: int = 4,
    device: str = "cpu",
    output_checkpoint: str = "checkpoints/modern_chat_finetuned.pt",
) -> ModernTransformer:
    """Supervised fine-tuning loop with prompt loss masking."""
    model.train()
    dataset = InstructionDataset(dataset_path, tokenizer, block_size=model.config.block_size)
    if len(dataset) == 0:
        raise ValueError(f"No valid instruction pairs found in {dataset_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"Finetuning model on {len(dataset)} instruction pairs for {steps} steps...")

    for step in range(1, steps + 1):
        x, y = dataset.sample_batch(batch_size, device=device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == steps:
            print(f"Finetune Step {step:3d}/{steps} | Loss: {loss.item():.4f}")

    save_checkpoint(model, output_checkpoint, step=steps, val_loss=loss.item())
    print(f"Finetuned checkpoint saved to {output_checkpoint}")
    return model

