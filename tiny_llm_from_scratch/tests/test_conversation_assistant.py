"""
Automated Conversation Assistant Verification Suite
Tests the 14 core conversation requirements:
A. Greetings: "Hi", "Hello", "Hey" -> Short natural greetings without novel continuations
B. Small talk: "How are you?", "Thanks", "Goodbye"
C. Simple factual questions: "What is Python?", "Explain gravity in one sentence."
D. Follow-ups: "Give an example", "Explain that more simply."
E. Coding: "Write a Python function that adds two numbers."
F. Mathematics: "What is 17 multiplied by 19?"
G. Ambiguity: "Fix this" with no content supplied -> Asks focused clarification
H. Insufficient knowledge: Admits limitation honestly without hallucination
I. Repetition & runaway output prevention
J. Conversation history continuity
K. Output formatting (no duplicate prompt, no novel leak)
L. Model loading & checkpoint preservation
M. No vendor branding in outputs
"""
import os
import sys
import unittest
from pathlib import Path
import re
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.orchestrator import ReasoningOrchestrator
from src.tools.builtin import solve_math_problem
from src.tools.code_engine import CodingAssistant, SafeCodeSandbox
from src.generation.template import ChatTemplate, ChatMessage
from src.generation.artifacts import ArtifactEngine


class TestConversationAssistant(unittest.TestCase):
    """Test suite for Conversational Assistant Upgrade."""

    def setUp(self):
        self.orch = ReasoningOrchestrator()

    # A. Greetings ("Hi", "Hello", "Hey")
    def test_A_greetings(self):
        for greeting in ["Hi", "Hello", "Hey", "good morning", "Hi!"]:
            prep = self.orch.prepare_reasoning(greeting)
            self.assertEqual(prep.intent, "greeting", f"Failed for {greeting}")
            self.assertIsNotNone(prep.direct_answer)
            # Must be short, polite, and not contain book / novel text
            self.assertIn("help you", prep.direct_answer.lower())
            self.assertLess(len(prep.direct_answer), 100)
            self.assertNotIn("prince", prep.direct_answer.lower())
            self.assertNotIn("chapter", prep.direct_answer.lower())

    # B. Small talk ("How are you?", "Thanks", "Goodbye")
    def test_B_small_talk(self):
        # How are you?
        prep_how = self.orch.prepare_reasoning("How are you?")
        self.assertEqual(prep_how.intent, "small-talk")
        self.assertIn("ready to help", prep_how.direct_answer)

        # Thanks
        prep_thanks = self.orch.prepare_reasoning("Thanks")
        self.assertEqual(prep_thanks.intent, "thanks")
        self.assertIn("welcome", prep_thanks.direct_answer.lower())

        # Goodbye
        prep_bye = self.orch.prepare_reasoning("Goodbye")
        self.assertEqual(prep_bye.intent, "goodbye")
        self.assertIn("goodbye", prep_bye.direct_answer.lower())

    # C. Simple factual questions
    def test_C_factual_questions(self):
        # What is Python?
        prep_python = self.orch.prepare_reasoning("What is Python?")
        self.assertIn(prep_python.intent, ["factual", "document-based"])
        self.assertIsNotNone(prep_python.direct_answer)
        self.assertIn("programming language", prep_python.direct_answer.lower())

        # Explain gravity in one sentence.
        prep_gravity = self.orch.prepare_reasoning("Explain gravity in one sentence.")
        self.assertIn(prep_gravity.intent, ["factual", "document-based"])
        self.assertIsNotNone(prep_gravity.direct_answer)
        self.assertIn("mass", prep_gravity.direct_answer.lower())

    # D. Follow-ups using conversation history
    def test_D_follow_ups(self):
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language known for its clean syntax."},
        ]
        prep_example = self.orch.prepare_reasoning("Give an example", conversation_history=history)
        self.assertEqual(prep_example.intent, "follow-up")
        self.assertIn("def ", prep_example.direct_answer)

        prep_simple = self.orch.prepare_reasoning("Explain that more simply", conversation_history=history)
        self.assertEqual(prep_simple.intent, "follow-up")
        self.assertIn("simple terms", prep_simple.direct_answer.lower())

    # E. Coding: "Write a Python function that adds two numbers."
    def test_E_coding_addition(self):
        prep = self.orch.prepare_reasoning("Write a Python function that adds two numbers.")
        self.assertEqual(prep.intent, "programming")
        self.assertIsNotNone(prep.direct_answer)
        self.assertIn("def add", prep.direct_answer)
        self.assertIn("return a + b", prep.direct_answer)

        # Test execution in sandbox
        sandbox_res = SafeCodeSandbox.execute_snippet("def add(a, b): return a + b\nprint(add(10, 20))")
        self.assertTrue(sandbox_res["success"])
        self.assertEqual(sandbox_res["stdout"], "30")

    # F. Mathematics: "What is 17 multiplied by 19?"
    def test_F_mathematics_word_arithmetic(self):
        prep = self.orch.prepare_reasoning("What is 17 multiplied by 19?")
        self.assertEqual(prep.intent, "mathematical")
        self.assertIsNotNone(prep.direct_answer)
        self.assertIn("323", prep.direct_answer)

        res = solve_math_problem("17 multiplied by 19")
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], 323)

    # G. Ambiguity: "Fix this" with no content supplied
    def test_G_ambiguity_clarification(self):
        prep = self.orch.prepare_reasoning("Fix this")
        self.assertEqual(prep.intent, "ambiguous")
        self.assertIn("please provide the code", prep.direct_answer.lower())

        # Incomplete Python expression debugging
        prep_debug = self.orch.prepare_reasoning("Fix this Python expression: print(1 + )")
        self.assertEqual(prep_debug.intent, "debugging")
        self.assertIn("incomplete", prep_debug.direct_answer.lower())

    # H. Insufficient knowledge handling
    def test_H_insufficient_knowledge(self):
        prep = self.orch.prepare_reasoning("What was the exact price of an apple on Mars?")
        self.assertEqual(prep.intent, "insufficient-knowledge")
        self.assertIn("do not have sufficient information", prep.direct_answer.lower())

    # I. Clean output formatting and stop sequences
    def test_I_output_formatting(self):
        chat_template = ChatTemplate()
        raw_model_stream = "Here is the answer.\n\n### User:\nNext question?\n### Assistant:\nFake turn"
        cleaned = chat_template.extract_assistant_response(raw_model_stream)
        self.assertEqual(cleaned, "Here is the answer.")

    # J. Conversation history template
    def test_J_conversation_history_template(self):
        messages = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello! How can I help you?"),
            ChatMessage(role="user", content="What is 5 + 5?"),
        ]
        template = ChatTemplate()
        formatted = template.format(messages)
        self.assertIn("### User:\nHi", formatted)
        self.assertIn("### Assistant:\nHello!", formatted)
        self.assertIn("### User:\nWhat is 5 + 5?", formatted)
        self.assertTrue(formatted.endswith("### Assistant:\n"))

    # K. Checkpoints preserved
    def test_K_checkpoint_preservation(self):
        ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
        self.assertTrue(ckpt_dir.is_dir())
        master_ckpt = ckpt_dir / "modern_master_v3_best.pt"
        self.assertTrue(master_ckpt.is_file(), f"Master checkpoint missing: {master_ckpt}")
        self.assertGreater(master_ckpt.stat().st_size, 10_000_000)

    # L. No vendor branding in responses or artifact cards
    def test_L_no_vendor_branding(self):
        for prompt in ["Hi", "What is Python?", "What is 17 multiplied by 19?"]:
            prep = self.orch.prepare_reasoning(prompt)
            answer = prep.direct_answer or ""
            thinking = prep.thinking or ""
            for vendor in ["claude", "chatgpt", "antigravity", "qwen", "gemma"]:
                self.assertNotIn(vendor, answer.lower(), f"Found vendor {vendor} in answer for {prompt}")
                self.assertNotIn(vendor, thinking.lower(), f"Found vendor {vendor} in thinking for {prompt}")

        # Artifact card
        sample_code = "```python\ndef foo(): pass\n```"
        arts = ArtifactEngine.extract_artifacts(sample_code)
        self.assertEqual(len(arts), 1)
        card = ArtifactEngine.format_artifact_display(arts[0])
        self.assertIn("[Artifact:", card)
        for vendor in ["claude", "chatgpt", "antigravity"]:
            self.assertNotIn(vendor, card.lower())


if __name__ == "__main__":
    unittest.main()
