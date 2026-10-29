from dataclasses import dataclass, field
from typing import Generator, List, Optional, Set, Union
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.90
    repetition_penalty: float = 1.15
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_new_tokens: int = 150
    stop_tokens: List[int] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)


class TextGenerator:
    """Robust autoregressive text generation engine with KV caching, streaming, and penalties."""

    def __init__(self, model, tokenizer: Tokenizer, default_config: Optional[SamplingConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = default_config or SamplingConfig()

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_ids: Union[List[int], torch.Tensor],
        config: Optional[SamplingConfig] = None,
        device: str = "cpu",
    ) -> Generator[str, None, None]:
        cfg = config or self.config
        self.model.eval()

        if isinstance(prompt_ids, list):
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        else:
            idx = prompt_ids.to(device)
            if idx.dim() == 1:
                idx = idx.unsqueeze(0)

        if hasattr(self.model, "block_size"):
            block_size = self.model.block_size
        elif hasattr(self.model, "config") and hasattr(self.model.config, "block_size"):
            block_size = self.model.config.block_size
        else:
            block_size = 256

        generated_tokens = []
        token_frequencies = {}

        # Pre-populate token frequencies from prompt for repetition penalties
        for t in idx[0].tolist():
            token_frequencies[t] = token_frequencies.get(t, 0) + 1

        # Check for Key-Value Cache capability
        use_kv = hasattr(self.model, "create_kv_caches")
        kv_caches = self.model.create_kv_caches() if use_kv else None

        current_pos = 0
        logits = None

        if use_kv:
            # Prefill phase: evaluate initial prompt in memory-safe chunks (supports up to 1M context)
            prompt_len = idx.shape[1]
            chunk_size = getattr(getattr(self.model, "config", None), "chunk_size", 512) or 512
            for start in range(0, prompt_len, chunk_size):
                end = min(start + chunk_size, prompt_len)
                chunk = idx[:, start:end]
                logits, _ = self.model(chunk, kv_caches=kv_caches, start_pos=start)
            logits = logits[:, -1, :].clone()
            current_pos = prompt_len
            next_token = None

        for step in range(cfg.max_new_tokens):
            if not use_kv:
                idx_cond = idx[:, -block_size:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :].clone()
            elif step > 0:
                # O(1) step inference via KV cache
                logits, _ = self.model(next_token, kv_caches=kv_caches, start_pos=current_pos)
                logits = logits[:, -1, :].clone()
                current_pos += 1

            # Check for NaN / Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

            # Repetition penalty
            if cfg.repetition_penalty > 1.0:
                for token_id in token_frequencies:
                    if token_id < logits.size(-1):
                        if logits[0, token_id] < 0:
                            logits[0, token_id] *= cfg.repetition_penalty
                        else:
                            logits[0, token_id] /= cfg.repetition_penalty

            # Frequency & Presence penalty
            if cfg.frequency_penalty > 0.0 or cfg.presence_penalty > 0.0:
                for token_id, freq in token_frequencies.items():
                    if token_id < logits.size(-1):
                        logits[0, token_id] -= (cfg.frequency_penalty * freq) + (cfg.presence_penalty * (1 if freq > 0 else 0))

            # Temperature
            temp = max(cfg.temperature, 1e-5)
            logits = logits / temp

            # Top-K filtering
            if cfg.top_k > 0:
                k = min(cfg.top_k, logits.size(-1))
                val, _ = torch.topk(logits, k)
                logits[logits < val[:, [-1]]] = -float("Inf")

            # Top-P (Nucleus) filtering
            if 0.0 < cfg.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > cfg.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            # Probability distribution & sampling
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()

            # Check stop tokens
            if cfg.stop_tokens and token_id in cfg.stop_tokens:
                break

            idx = torch.cat((idx, next_token), dim=1)
            generated_tokens.append(token_id)
            token_frequencies[token_id] = token_frequencies.get(token_id, 0) + 1

            decoded_text = self.tokenizer.decode([token_id])

            # Check stop strings
            stop_hit = False
            for s in cfg.stop_strings:
                if s in decoded_text:
                    stop_hit = True
                    break
            if stop_hit:
                break

            yield decoded_text

    def generate(
        self,
        prompt: str,
        config: Optional[SamplingConfig] = None,
        device: str = "cpu",
    ) -> str:
        prompt_ids = self.tokenizer.encode(prompt).ids
        pieces = list(self.generate_stream(prompt_ids, config=config, device=device))
        return "".join(pieces)
