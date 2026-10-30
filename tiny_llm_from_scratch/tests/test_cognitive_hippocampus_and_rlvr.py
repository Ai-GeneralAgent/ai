"""
Unit Tests for Cognitive Tiered Memory (AI Hippocampus) & RL with Verifiable Rewards (RLVR)
"""
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.hippocampus import HippocampusMemory, DurableFact
from src.reasoning.rlvr import DeterministicVerifier, SelfVerifiedDistillationManager


class TestCognitiveHippocampus(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_hippocampus.db"
        self.mem = HippocampusMemory(db_path=self.db_path, working_window=4)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_working_memory_window(self):
        for i in range(6):
            self.mem.add_to_working_memory("user", f"message {i}")
        ctx = self.mem.get_working_context()
        self.assertEqual(len(ctx), 4)
        self.assertEqual(ctx[-1]["content"], "message 5")

    def test_store_and_recall_facts(self):
        stored = self.mem.store_fact(
            "User's name is Yuvraj Thakur.",
            category="profile",
            confidence=1.0,
            tags=["name", "user", "yuvraj", "creator"],
        )
        self.assertTrue(stored)

        # Associative Recall
        recalled = self.mem.recall_relevant_memories("Who is Yuvraj?", top_k=2)
        self.assertGreater(len(recalled), 0)
        self.assertIn("Yuvraj Thakur", recalled[0].content)
        self.assertEqual(recalled[0].recall_count, 1)

    def test_auto_fact_extraction(self):
        extracted = self.mem.extract_and_store_facts("My name is Yuvraj Thakur and I prefer short natural answers.")
        self.assertGreater(len(extracted), 0)
        recalled = self.mem.recall_relevant_memories("What is my name?")
        self.assertGreater(len(recalled), 0)
        self.assertIn("Yuvraj Thakur", recalled[0].content)

    def test_continual_training_export(self):
        self.mem.store_fact("Project rule: Always use D: drive, strictly avoid writing to C: drive.", category="rule")
        pairs = self.mem.export_memories_for_training()
        self.assertGreater(len(pairs), 0)
        self.assertIn("instruction", pairs[0])
        self.assertIn("response", pairs[0])


class TestDeterministicVerifier(unittest.TestCase):
    def test_math_verification_correct(self):
        res = DeterministicVerifier.verify_math_expression("17 * 19", "The answer is 323")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.reward_score, 1.0)

    def test_math_verification_incorrect(self):
        res = DeterministicVerifier.verify_math_expression("17 * 19", "The answer is 999")
        self.assertFalse(res.is_valid)
        self.assertEqual(res.reward_score, 0.0)

    def test_python_code_verification(self):
        code = "def add(a, b):\n    return a + b"
        assertions = [
            "assert add(2, 3) == 5",
            "assert add(-1, 1) == 0",
        ]
        res = DeterministicVerifier.verify_python_code(code, assertions)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.reward_score, 1.0)


class TestSelfVerifiedDistillation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_path = Path(self.temp_dir.name) / "test_rlvr.jsonl"
        self.manager = SelfVerifiedDistillationManager(data_output_path=self.out_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_candidate_selection_and_logging(self):
        candidates = [
            "I think 25 * 4 is 95.",
            "Calculation: 25 * 4 = 100.",
            "Maybe it is 105.",
        ]
        best_cand, best_rew, _ = self.manager.evaluate_candidates(
            problem="25 * 4",
            candidates=candidates,
            verifier_type="math",
            verification_target="25 * 4",
        )
        self.assertIsNotNone(best_cand)
        self.assertEqual(best_rew, 1.0)
        self.assertIn("100", best_cand)

        # Check that verified sample was logged
        self.assertTrue(self.out_path.is_file())
        lines = [json.loads(l) for l in self.out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()

