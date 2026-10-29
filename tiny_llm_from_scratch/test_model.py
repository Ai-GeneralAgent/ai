import unittest
import torch
from model import TinyGPT


class TinyGPTTests(unittest.TestCase):
    def test_forward_shape_and_loss(self):
        model = TinyGPT(vocab_size=20, block_size=16, n_embd=32, n_head=4, n_layer=2)
        x = torch.randint(0, 20, (2, 16))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 16, 20))
        self.assertTrue(torch.isfinite(loss).item())

    def test_generation_length(self):
        model = TinyGPT(vocab_size=10, block_size=8, n_embd=16, n_head=4, n_layer=1)
        start = torch.zeros((1, 3), dtype=torch.long)
        result = model.generate(start, max_new_tokens=5)
        self.assertEqual(result.shape[1], 8)


if __name__ == "__main__":
    unittest.main()
