"""Regression tests for fixes: RAG import, bounded generation positions, deterministic eval, and Muon optimizer."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import ModelConfig
from src.models.transformer import ModernTransformer
from src.generation.sampler import TextGenerator, SamplingConfig
from src.training.trainer import Trainer, TrainingConfig
from src.data.token_cache import CompactTokenCache


class _Spy(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.config, self.highest = inner, inner.config, 0

    def create_kv_caches(self):
        return self.inner.create_kv_caches()

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        self.highest = max(self.highest, start_pos + idx.shape[1])
        return self.inner(idx, targets, kv_caches, start_pos)


class _DummyTok:
    def decode(self, ids):
        return " ".join(map(str, ids))


class UpgradeTests(unittest.TestCase):
    def test_rag_module_imports(self):
        from src.rag import LocalRAGPipeline  # noqa: F401  (used to raise NameError: Any)

    def test_generation_never_exceeds_block_size(self):
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=200, block_size=32, n_embd=32, n_head=4, n_layer=2, max_seq_len=1024)
        spy = _Spy(ModernTransformer(cfg).eval())
        gen = TextGenerator(spy, _DummyTok())
        sc = SamplingConfig(max_new_tokens=200, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
        out = list(gen.generate_stream(list(range(10)), config=sc))
        self.assertEqual(len(out), 200)
        self.assertLessEqual(spy.highest, cfg.block_size)


class MuonTests(unittest.TestCase):
    def _make(self):
        torch.manual_seed(0)
        tmp = Path(tempfile.mkdtemp())
        arr = (np.arange(6000) % 17).astype(np.uint16)  # learnable pattern
        f = tmp / "t.u16"
        arr.tofile(f)
        cache = CompactTokenCache(f, len(arr))
        cfg = ModelConfig(vocab_size=200, block_size=32, n_embd=32, n_head=4, n_layer=2)
        tc = TrainingConfig(
            max_steps=20,
            learning_rate=3e-3,
            warmup_steps=2,
            batch_size=4,
            eval_interval=10**9,
            output_dir=str(tmp),
            device="cpu",
            optimizer="muon",
        )
        return Trainer(ModernTransformer(cfg), tc, cache, tmp / "tok.json"), cache

    def test_muon_runs_and_state_roundtrips(self):
        tr, cache = self._make()
        before = tr.evaluate()
        with contextlib.redirect_stdout(io.StringIO()):
            tr.train()
        self.assertLess(tr.evaluate(), before - 1.0)
        state = tr.optimizer.state_dict()
        tr.optimizer.load_state_dict(state)
        self.assertGreaterEqual(len(tr.optimizer.param_groups), 2)
        cache.close()


class MultiTokenPredictionTests(unittest.TestCase):
    def test_mtp_architecture_and_forward_pass(self):
        torch.manual_seed(42)
        cfg = ModelConfig(vocab_size=100, block_size=32, n_embd=32, n_head=4, n_layer=2, n_predict_tokens=3)
        model = ModernTransformer(cfg)
        self.assertIsNotNone(model.extra_heads)
        self.assertEqual(len(model.extra_heads), 2)

        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))

        # Curriculum k=1
        logits_1, loss_1 = model(x, y, curriculum_k=1)
        self.assertEqual(logits_1.shape, (2, 16, 100))
        self.assertIsNotNone(loss_1)

        # Curriculum k=2
        _, loss_2 = model(x, y, curriculum_k=2)
        self.assertIsNotNone(loss_2)

        # Curriculum k=3
        _, loss_3 = model(x, y, curriculum_k=3)
        self.assertIsNotNone(loss_3)

        # Test gradient propagation
        loss_3.backward()
        for head in model.extra_heads:
            self.assertIsNotNone(head.weight.grad)


if __name__ == "__main__":
    unittest.main()
