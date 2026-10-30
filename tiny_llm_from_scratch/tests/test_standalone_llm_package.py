"""
Unit Tests for Standalone Vendored LLM Package
"""
import sys
from pathlib import Path
import unittest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm import (
    LLMConfig,
    LLMForCausalLM,
    LLMTokenizer,
    load_llm_checkpoint,
)


class TestStandaloneLLMPackage(unittest.TestCase):
    def test_config_presets(self):
        cfg_tiny = LLMConfig.preset_tiny(vocab_size=1000)
        self.assertEqual(cfg_tiny.model_type, "llm")
        self.assertEqual(cfg_tiny.hidden_size, 192)
        self.assertEqual(cfg_tiny.num_hidden_layers, 3)

        d = cfg_tiny.to_dict()
        restored = LLMConfig.from_dict(d)
        self.assertEqual(restored.hidden_size, 192)

    def test_model_forward_and_loss(self):
        cfg = LLMConfig.preset_tiny(vocab_size=500)
        model = LLMForCausalLM(cfg)
        model.eval()

        input_ids = torch.randint(0, 500, (2, 8))
        labels = input_ids.clone()

        out = model(input_ids, labels=labels)
        self.assertIsNotNone(out.logits)
        self.assertEqual(out.logits.shape, (2, 8, 500))
        self.assertIsNotNone(out.loss)
        self.assertGreater(out.loss.item(), 0.0)

    def test_kv_cache_generation(self):
        cfg = LLMConfig.preset_tiny(vocab_size=300)
        model = LLMForCausalLM(cfg)
        model.eval()

        input_ids = torch.randint(0, 300, (1, 4))
        generated = model.generate(input_ids, max_new_tokens=6, temperature=0.7)
        self.assertEqual(generated.shape, (1, 10))

    def test_tokenizer_chat_template(self):
        tok_path = ROOT_DIR / "tiny_llm_from_scratch" / "tokenizer_bpe.json"
        if tok_path.is_file():
            tok = LLMTokenizer(tok_path)
            msgs = [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            tmpl = tok.apply_chat_template(msgs, add_generation_prompt=True)
            self.assertIn("<|im_start|>user\nHello!<|im_end|>", tmpl)
            self.assertIn("<|im_start|>assistant\nHi there!<|im_end|>", tmpl)
            self.assertTrue(tmpl.endswith("<|im_start|>assistant\n"))


if __name__ == "__main__":
    unittest.main()

