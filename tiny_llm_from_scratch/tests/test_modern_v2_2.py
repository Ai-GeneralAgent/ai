"""
Tests for Modern LLM v2.2 Upgrades:
- 1,048,576 (1M) Context Window & YaRN / Dynamic NTK RoPE
- StreamingLLM Attention Sinks & Sliding Window KV Cache
- Chunked Prefill for Ultra-Long Sequences
- DuckDuckGo Live Web Search Engine
- Claude-Style Artifact Extraction & Rendering
- Web Search Intent Classification in Reasoning Orchestrator
"""
import unittest
import torch

from src.models.config import ModelConfig
from src.models.transformer import ModernTransformer, KVCache, precompute_rope_freqs
from src.generation.sampler import TextGenerator, SamplingConfig
from src.generation.artifacts import ClaudeArtifactEngine, ClaudeArtifact
from src.tools.web_search import DuckDuckGoSearchEngine
from src.reasoning.orchestrator import ReasoningOrchestrator


class TestModernV22Upgrades(unittest.TestCase):
    """Test suite for 1M context, DuckDuckGo search, and Claude capabilities."""

    def test_01_preset_1m_config(self):
        cfg = ModelConfig.preset_1m(vocab_size=10000)
        self.assertEqual(cfg.max_seq_len, 1_048_576)
        self.assertEqual(cfg.rope_scaling_type, "yarn")
        self.assertEqual(cfg.rope_theta, 1_000_000.0)
        self.assertEqual(cfg.sink_tokens, 4)
        self.assertEqual(cfg.sliding_window, 4096)
        self.assertEqual(cfg.arch_version, "2.2.0")

    def test_02_yarn_rope_frequency_scaling(self):
        # Base length 256
        freqs_base = precompute_rope_freqs(
            dim=48, max_seq_len=256, theta=10000.0, scaling_type="yarn", original_max_len=256
        )
        self.assertEqual(freqs_base.shape, (256, 24))

        # Extrapolated to 10,000 positions with YaRN
        freqs_yarn = precompute_rope_freqs(
            dim=48, max_seq_len=10000, theta=10000.0, scaling_type="yarn", original_max_len=256
        )
        self.assertEqual(freqs_yarn.shape, (10000, 24))
        self.assertFalse(torch.isnan(freqs_yarn).any())
        self.assertFalse(torch.isinf(freqs_yarn).any())

    def test_03_kv_cache_streaming_sinks(self):
        # Window size 8, sink size 2
        cache = KVCache(max_capacity=1_048_576, window_size=8, sink_size=2)
        # Push 20 tokens through cache in steps
        for step in range(20):
            k_step = torch.full((1, 4, 1, 48), float(step))
            v_step = torch.full((1, 4, 1, 48), float(step))
            k_all, v_all = cache.update(k_step, v_step)

        # Total stored should be sink_size + window_size = 2 + 8 = 10 tokens
        self.assertEqual(k_all.shape[2], 10)
        # Verify first 2 tokens are the sinks (step 0 and step 1)
        self.assertEqual(k_all[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(k_all[0, 0, 1, 0].item(), 1.0)
        # Verify last token is the latest token (step 19)
        self.assertEqual(k_all[0, 0, -1, 0].item(), 19.0)

    def test_04_model_1m_context_forward_step(self):
        cfg = ModelConfig(
            vocab_size=100,
            block_size=64,
            max_seq_len=1_048_576,
            n_embd=64,
            n_head=2,
            n_layer=2,
            pos_emb="rope",
            rope_scaling_type="yarn",
        )
        model = ModernTransformer(cfg)
        model.eval()

        # Step at sequence position 500,000 (well beyond standard 256/1024 base!)
        single_token = torch.tensor([[42]], dtype=torch.long)
        kv_caches = model.create_kv_caches()
        logits, _ = model(single_token, kv_caches=kv_caches, start_pos=500_000)
        self.assertEqual(logits.shape, (1, 1, 100))
        self.assertFalse(torch.isnan(logits).any())

    def test_05_duckduckgo_search_engine(self):
        results = DuckDuckGoSearchEngine.search("Albert Einstein", max_results=3)
        self.assertIsInstance(results, list)
        if results:  # If internet is reachable
            self.assertIn("title", results[0])
            self.assertIn("url", results[0])
            self.assertIn("snippet", results[0])
            formatted = DuckDuckGoSearchEngine.format_search_results("Albert Einstein", results)
            self.assertIn("DuckDuckGo Web Search Results", formatted)
            self.assertIn("http", formatted)

    def test_06_claude_artifact_extraction(self):
        text = (
            "Here is a code artifact:\n\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n\n"
            "And a diagram:\n\n"
            "```mermaid\n"
            "graph LR\n"
            "    A --> B\n"
            "```\n"
        )
        artifacts = ClaudeArtifactEngine.extract_artifacts(text)
        self.assertEqual(len(artifacts), 2)
        self.assertEqual(artifacts[0].artifact_type, "code")
        self.assertEqual(artifacts[0].language, "python")
        self.assertEqual(artifacts[1].artifact_type, "diagram")

        card = ClaudeArtifactEngine.format_artifact_display(artifacts[0])
        self.assertIn("Artifact:", card)
        self.assertIn("def add", card)

    def test_07_orchestrator_web_search_intent(self):
        orch = ReasoningOrchestrator()
        intent_1 = orch.classify_intent("/search quantum computing")
        self.assertEqual(intent_1, "web-search")

        intent_2 = orch.classify_intent("search duckduckgo for latest developments in transformers")
        self.assertEqual(intent_2, "web-search")

        prep = orch.prepare_reasoning("/search quantum computing")
        self.assertEqual(prep.intent, "web-search")
        self.assertIn("Web Search Dispatch", prep.thinking)


if __name__ == "__main__":
    unittest.main()
