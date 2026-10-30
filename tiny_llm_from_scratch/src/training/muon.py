"""Muon + AdamW hybrid optimizer (opt-in).

Muon orthogonalizes the momentum of 2D hidden-layer weight matrices with a
Newton-Schulz iteration. Embeddings, norms and other 1D parameters stay on AdamW.
References: Jordan et al. (2024) "Muon"; Liu et al. (2025) "Muon is Scalable for LLM Training".
"""
from typing import Any, Dict, Iterable, List

import torch


def newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class HybridMuonAdamW:
    """Drop-in optimizer: exposes param_groups / step / zero_grad / state_dict like torch optimizers."""

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        adam_groups: List[Dict[str, Any]],
        lr: float,
        muon_lr: float = 0.02,
        momentum: float = 0.95,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
    ):
        self.muon_params = list(muon_params)
        self.momentum = momentum
        self._buf = [torch.zeros_like(p) for p in self.muon_params]
        self._adamw = torch.optim.AdamW(adam_groups, lr=lr, betas=betas, eps=eps)
        self.muon_group: Dict[str, Any] = {
            "params": self.muon_params,
            "lr": muon_lr,
            "lr_scale": muon_lr / max(1e-9, lr),
        }
        # Shared dict objects: the LR scheduler edits param_groups and both optimizers see it.
        self.param_groups = [self.muon_group] + self._adamw.param_groups

    @torch.no_grad()
    def step(self):
        for p, buf in zip(self.muon_params, self._buf):
            if p.grad is None:
                continue
            buf.mul_(self.momentum).add_(p.grad)
            g = p.grad.add(buf, alpha=self.momentum)  # Nesterov
            u = newton_schulz5(g) * max(1.0, p.size(0) / p.size(1)) ** 0.5
            p.add_(u.to(p.dtype), alpha=-self.muon_group["lr"])
        self._adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        for p in self.muon_params:
            p.grad = None
        self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"adamw": self._adamw.state_dict(), "muon_buf": [b.clone() for b in self._buf]}

    def load_state_dict(self, state):
        self._adamw.load_state_dict(state["adamw"])
        for b, s in zip(self._buf, state["muon_buf"]):
            b.copy_(s)

