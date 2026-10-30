"""
Unit Tests for Claude & ChatGPT-Style Chat Memory Architecture:
1. Working context & rolling conversation buffer.
2. Context compaction with prior dialogue summary (Claude /compact style).
3. Automatic fact extraction from natural directives ("Remember that...", "My favorite...", "I work as...").
4. Associative recall & broad profile retrieval.
5. Persistent system memory injection (<durable_user_memory>).
6. Memory CRUD operations: list, delete, clear.
"""
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.hippocampus import HippocampusMemory, DurableFact


class TestClaudeChatGPTChatMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_chat_memory.db"
        self.mem = HippocampusMemory(db_path=self.db_path, working_window=6)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_explicit_remember_directive(self):
        extracted = self.mem.extract_and_store_facts("Remember that my dog's name is Rex.")
        self.assertGreater(len(extracted), 0)
        self.assertTrue(any("Rex" in f for f in extracted))

        recalled = self.mem.recall_relevant_memories("What is my dog's name?")
        self.assertGreater(len(recalled), 0)
        self.assertIn("Rex", recalled[0].content)

    def test_favorite_preference_extraction(self):
        extracted = self.mem.extract_and_store_facts("My favorite framework is PyTorch.")
        self.assertGreater(len(extracted), 0)
        self.assertTrue(any("PyTorch" in f for f in extracted))

        recalled = self.mem.recall_relevant_memories("Tell me about my favorite framework")
        self.assertGreater(len(recalled), 0)
        self.assertIn("PyTorch", recalled[0].content)

    def test_job_and_location_extraction(self):
        self.mem.extract_and_store_facts("I work as a machine learning engineer and I live in Tokyo.")
        all_facts = self.mem.list_all_facts()
        contents = [f.content for f in all_facts]
        self.assertTrue(any("machine learning engineer" in c for c in contents))
        self.assertTrue(any("Tokyo" in c for c in contents))

    def test_project_extraction(self):
        self.mem.extract_and_store_facts("Our project is building an autonomous local reasoning LLM.")
        all_facts = self.mem.list_all_facts()
        self.assertTrue(any("autonomous local reasoning LLM" in f.content for f in all_facts))

    def test_broad_memory_query(self):
        self.mem.store_fact("User's name is Yuvraj Thakur.", category="profile")
        self.mem.store_fact("User's favorite drink is green tea.", category="preference")
        
        recalled = self.mem.recall_relevant_memories("What do you remember about me?")
        self.assertGreater(len(recalled), 0)
        recalled_contents = [f.content for f in recalled]
        self.assertTrue(any("Yuvraj Thakur" in c for c in recalled_contents))

    def test_format_system_memory_context(self):
        self.mem.store_fact("User's name is Yuvraj Thakur.", category="profile")
        self.mem.store_fact("User preference: Always write clean type-annotated Python.", category="preference")
        
        sys_block = self.mem.format_system_memory_context("Python code")
        self.assertIn("<durable_user_memory>", sys_block)
        self.assertIn("Yuvraj Thakur", sys_block)
        self.assertIn("clean type-annotated Python", sys_block)
        self.assertIn("</durable_user_memory>", sys_block)

    def test_memory_crud_operations(self):
        self.mem.store_fact("Fact 1: User prefers Linux.", category="preference")
        self.mem.store_fact("Fact 2: Project uses SQLite.", category="project")
        self.mem.store_fact("Fact 3: Temporary note.", category="knowledge")
        
        all_facts = self.mem.list_all_facts()
        self.assertEqual(len(all_facts), 3)

        deleted = self.mem.delete_fact("Temporary note")
        self.assertEqual(deleted, 1)
        self.assertEqual(len(self.mem.list_all_facts()), 2)

        cleared = self.mem.clear_all_facts()
        self.assertEqual(cleared, 2)
        self.assertEqual(len(self.mem.list_all_facts()), 0)

    def test_conversation_context_compaction(self):
        messages = [
            {"role": "user", "content": f"User question {i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"Assistant response {i}"}
            for i in range(16)
        ]
        recent, summary_note = HippocampusMemory.compact_conversation_history(messages, keep_recent=6)
        self.assertEqual(len(recent), 6)
        self.assertIsNotNone(summary_note)
        self.assertIn("Prior Conversation Context Summary (10 earlier turns)", summary_note)
        self.assertEqual(recent[0]["content"], "User question 10")
        self.assertEqual(recent[-1]["content"], "Assistant response 15")


if __name__ == "__main__":
    unittest.main()
