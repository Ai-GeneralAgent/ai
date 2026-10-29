"""
Direct Preference Optimization (DPO) Engine
Reference-free preference tuning without separate reward models.
Implements Rafailov et al. (2023) mathematical objective for preference alignment:
L_DPO(theta; ref) = - E_(x, y_w, y_l) [ log sigma( beta * log(pi_theta(y_w|x)/pi_ref(y_w|x)) - beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)) ) ]
"""
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from ..models.transformer import ModernTransformer
from ..models.checkpoint import save_checkpoint


class DPODataset:
    """Prepares pairwise (prompt, chosen, rejected) examples for preference alignment."""

    def __init__(self, jsonl_path: Union[str, Path], tokenizer: Tokenizer, block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pairs: List[Dict[str, torch.Tensor]] = []

        p = Path(jsonl_path)
        if not p.is_file():
            raise FileNotFoundError(f"DPO dataset not found: {p.resolve()}")

        with p.open("r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    prompt = obj.get("prompt") or obj.get("instruction") or ""
                    chosen = obj.get("chosen") or obj.get("accepted") or obj.get("preferred") or ""
                    rejected = obj.get("rejected") or obj.get("dispreferred") or ""
                    if prompt and chosen and rejected:
                        self._process_pair(prompt, chosen, rejected)
                except Exception:
                    continue

    def _process_pair(self, prompt: str, chosen: str, rejected: str):
        prompt_text = f"### User:\n{prompt}\n\n### Assistant:\n"
        prompt_ids = self.tokenizer.encode(prompt_text).ids

        def _encode_completion(resp: str) -> Tuple[torch.Tensor, torch.Tensor]:
            full = f"{prompt_text}{resp}\n"
            full_ids = self.tokenizer.encode(full).ids[: self.block_size + 1]
            x_ids = full_ids[:-1]
            y_ids = full_ids[1:]

            prompt_len = min(len(prompt_ids) - 1, len(y_ids))
            targets = list(y_ids)
            for i in range(prompt_len):
                targets[i] = -100  # Mask prompt tokens

            return torch.tensor(x_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

        x_w, y_w = _encode_completion(chosen)
        x_l, y_l = _encode_completion(rejected)

        self.pairs.append({
            "chosen_x": x_w,
            "chosen_y": y_w,
            "rejected_x": x_l,
            "rejected_y": y_l,
        })

    def __len__(self) -> int:
        return len(self.pairs)

    def sample_batch(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        indices = torch.randint(0, len(self.pairs), (batch_size,))
        items = [self.pairs[i] for i in indices]

        def _collate(key_x: str, key_y: str) -> Tuple[torch.Tensor, torch.Tensor]:
            max_len = max(it[key_x].size(0) for it in items)
            bx, by = [], []
            for it in items:
                pad = max_len - it[key_x].size(0)
                bx.append(F.pad(it[key_x], (0, pad), value=0) if pad > 0 else it[key_x])
                by.append(F.pad(it[key_y], (0, pad), value=-100) if pad > 0 else it[key_y])
            return torch.stack(bx).to(device), torch.stack(by).to(device)

        cx, cy = _collate("chosen_x", "chosen_y")
        rx, ry = _collate("rejected_x", "rejected_y")

        return {
            "chosen_x": cx,
            "chosen_y": cy,
            "rejected_x": rx,
            "rejected_y": ry,
        }


def compute_sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes sum of log probabilities for non-masked tokens (-100)."""
    # logits: (B, T, V), labels: (B, T)
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather target log probs
    labels_clamped = labels.clone()
    mask = labels != -100
    labels_clamped[~mask] = 0

    per_token_log_probs = torch.gather(log_probs, dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
    per_token_log_probs = per_token_log_probs * mask.float()
    return per_token_log_probs.sum(dim=-1)


def dpo_loss(
    pi_chosen_log_probs: torch.Tensor,
    pi_rejected_log_probs: torch.Tensor,
    ref_chosen_log_probs: torch.Tensor,
    ref_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes DPO loss and implicit rewards:
    L_DPO = - log sigma ( beta * (log pi(y_w|x) - log ref(y_w|x)) - beta * (log pi(y_l|x) - log ref(y_l|x)) )
    """
    pi_logratios = pi_chosen_log_probs - pi_rejected_log_probs
    ref_logratios = ref_chosen_log_probs - ref_rejected_log_probs

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        chosen_rewards = beta * (pi_chosen_log_probs - ref_chosen_log_probs)
        rejected_rewards = beta * (pi_rejected_log_probs - ref_rejected_log_probs)

    return loss, chosen_rewards.mean(), rejected_rewards.mean()


class DPOTrainer:
    """Production Direct Preference Optimization Trainer."""

    def __init__(
        self,
        model: ModernTransformer,
        dataset: DPODataset,
        ref_model: Optional[ModernTransformer] = None,
        beta: float = 0.1,
        lr: float = 5e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.beta = beta
        self.device = device

        if ref_model is None:
            # Create frozen reference policy
            self.ref_model = copy.deepcopy(model).to(device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            self.ref_model = ref_model.to(device)
            self.ref_model.eval()

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )

    def train_step(self, batch_size: int = 4) -> Dict[str, float]:
        self.model.train()
        batch = self.dataset.sample_batch(batch_size, device=self.device)

        # Policy forward passes
        logits_chosen, _ = self.model(batch["chosen_x"])
        logits_rejected, _ = self.model(batch["rejected_x"])

        pi_chosen_lp = compute_sequence_log_probs(logits_chosen, batch["chosen_y"])
        pi_rejected_lp = compute_sequence_log_probs(logits_rejected, batch["rejected_y"])

        # Reference forward passes (no grad)
        with torch.no_grad():
            ref_logits_chosen, _ = self.ref_model(batch["chosen_x"])
            ref_logits_rejected, _ = self.ref_model(batch["rejected_x"])

            ref_chosen_lp = compute_sequence_log_probs(ref_logits_chosen, batch["chosen_y"])
            ref_rejected_lp = compute_sequence_log_probs(ref_logits_rejected, batch["rejected_y"])

        loss, r_chosen, r_rejected = dpo_loss(
            pi_chosen_lp, pi_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=self.beta
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward_chosen": r_chosen.item(),
            "reward_rejected": r_rejected.item(),
            "reward_margin": (r_chosen - r_rejected).item(),
        }
