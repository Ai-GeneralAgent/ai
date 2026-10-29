import torch
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def forward(self, x):
        n = x.size(1)
        z = self.ln1(x)
        y, _ = self.attn(
            z, z, z,
            attn_mask=self.mask[:n, :n],
            need_weights=False,
        )
        x = x + self.dropout(y)
        x = x + self.ff(self.ln2(x))
        return x


class TinyBPEGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        n_embd=192,
        n_head=4,
        n_layer=3,
        dropout=0.1,
    ):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        _, length = idx.shape
        if length > self.block_size:
            raise ValueError(
                f"Input length {length} exceeds context {self.block_size}"
            )

        positions = torch.arange(length, device=idx.device)
        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)[None, :, :]
        )

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_ln(x))
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 0.01)

            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
