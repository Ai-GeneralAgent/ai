"""
End-to-End Integration Tests for Claude/ChatGPT Chat Memory Architecture:
1. Multi-turn dialogue working context preservation.
2. Context compaction with prior summary note.
3. Natural memory directives ("Remember that...", "My favorite...", "I work as...").
4. Associative recall & broad profile retrieval.
5. System prompt memory block injection (<durable_user_memory>).
6. Memory CRUD management (list, add, forget, clear).
"""
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.hippocampus import HippocampusMemory, DurableFact
from src.memory.conversation_db import ConversationDatabase
from src.reasoning.orchestrator import ReasoningOrchestrator


class TestChatMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.hippo_db = Path(self.temp_dir.name) / "test_hippocampus.db"
        self.conv_db = Path(self.temp_dir.name) / "test_conversation.db"
        
        self.hippocampus = HippocampusMemory(db_path=self.hippo_db, working_window=6)
        self.conv_memory = ConversationDatabase(db_path=self.conv_db)
        self.session_id = "test_integration_session"
        self.conv_memory.create_session(self.session_id, "Integration Test Session")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multi_turn_dialogue_and_fact_extraction(self):
        # Turn 1: User introduces self and preference
        t1_user = "Hi! Remember that my name is Yuvraj and my favorite language is Python."
        extracted = self.hippocampus.extract_and_store_facts(t1_user, session_id=self.session_id)
        self.assertGreater(len(extracted), 0)
        self.conv_memory.add_message(self.session_id, "user", t1_user)
        self.hippocampus.log_turn(self.session_id, "user", t1_user)

        t1_asst = "Hello Yuvraj! It's fantastic to meet you. I've noted that Python is your favorite language."
        self.conv_memory.add_message(self.session_id, "assistant", t1_asst)
        self.hippocampus.log_turn(self.session_id, "assistant", t1_asst)

        # Turn 2: Verify memories are retrievable
        recalled = self.hippocampus.recall_relevant_memories("What is my name?")
        self.assertGreater(len(recalled), 0)
        self.assertTrue(any("Yuvraj" in r.content for r in recalled))

        recalled_lang = self.hippocampus.recall_relevant_memories("What language do I like?")
        self.assertGreater(len(recalled_lang), 0)
        self.assertTrue(any("Python" in r.content for r in recalled_lang))

        # Turn 3: Verify system memory block formatting (Claude/ChatGPT style)
        sys_block = self.hippocampus.format_system_memory_context("Python scripting", top_k=3)
        self.assertIn("<durable_user_memory>", sys_block)
        self.assertIn("Yuvraj", sys_block)
        self.assertIn("Python", sys_block)
        self.assertIn("</durable_user_memory>", sys_block)

    def test_context_compaction_on_long_session(self):
        # Simulate 14 dialogue turns
        for i in range(14):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Turn {i}: discussing project architecture milestone {i}."
            self.conv_memory.add_message(self.session_id, role, content)

        all_msgs = self.conv_memory.get_messages(self.session_id, limit=20)
        self.assertEqual(len(all_msgs), 14)

        compacted, summary_note = self.hippocampus.compact_conversation_history(all_msgs, keep_recent=6)
        self.assertEqual(len(compacted), 6)
        self.assertIsNotNone(summary_note)
        self.assertIn("Prior Conversation Context Summary (8 earlier turns)", summary_note)
        self.assertEqual(compacted[-1]["content"], "Turn 13: discussing project architecture milestone 13.")

    def test_memory_crud_workflow(self):
        # Add facts
        self.hippocampus.store_fact("User works on tiny LLM reasoning.", category="project")
        self.hippocampus.store_fact("User's timezone is UTC+5:30.", category="profile")
        self.hippocampus.store_fact("User dislikes boilerplate code.", category="preference")

        facts = self.hippocampus.list_all_facts()
        self.assertEqual(len(facts), 3)

        # Delete by keyword
        del_count = self.hippocampus.delete_fact("boilerplate")
        self.assertEqual(del_count, 1)
        self.assertEqual(len(self.hippocampus.list_all_facts()), 2)

        # Clear all
        clear_count = self.hippocampus.clear_all_facts()
        self.assertEqual(clear_count, 2)
        self.assertEqual(len(self.hippocampus.list_all_facts()), 0)


if __name__ == "__main__":
    unittest.main()
