"""
Unit tests for InternetLearnerEngine and Autonomous Web Ingestion.
Tests DuckDuckGo search integration, deep web browsing, fact distillation,
Hippocampus SQLite storage, Local RAG indexing, and continual training dataset export.
"""
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from src.tools.internet_learner import InternetLearnerEngine, LearnedKnowledgePackage
from src.tools.registry import ToolRegistry
from src.tools.builtin import register_builtin_tools
from src.reasoning.orchestrator import ReasoningOrchestrator


class TestInternetLearnerEngine(unittest.TestCase):
    """Test suite for autonomous internet learning and knowledge distillation."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent

    def test_determine_category(self):
        self.assertEqual(InternetLearnerEngine.determine_category("Python programming"), "coding")
        self.assertEqual(InternetLearnerEngine.determine_category("SQL injection exploit"), "cybersecurity")
        self.assertEqual(InternetLearnerEngine.determine_category("Stock market volatility"), "finance_stock_market")
        self.assertEqual(InternetLearnerEngine.determine_category("Quantum physics entanglement"), "science_research")

    def test_extract_facts_from_text(self):
        raw_text = (
            "Python is a versatile high-level programming language. "
            "It was developed by Guido van Rossum and released in 1991. "
            "The weather in Tokyo is sunny today. "
            "Python consists of a large standard library and clean syntax. "
            "Some random unrelated sentence that has nothing to do with the topic."
        )
        facts = InternetLearnerEngine.extract_facts_from_text("Python", raw_text, max_facts=3)
        self.assertGreater(len(facts), 0)
        self.assertTrue(any("programming language" in f.lower() for f in facts))

    @patch("src.tools.web_search.DuckDuckGoSearchEngine.search")
    @patch("src.tools.browser.WebBrowserEngine.browse_url")
    def test_learn_end_to_end_mock(self, mock_browse, mock_search):
        mock_search.return_value = [
            {
                "title": "Rust Language",
                "snippet": "Rust is a systems programming language focused on safety and speed.",
                "url": "https://www.rust-lang.org/",
                "source": "DuckDuckGo",
            }
        ]
        mock_browse.return_value = {
            "success": True,
            "url": "https://www.rust-lang.org/",
            "clean_text": "Rust is defined as a multi-paradigm language. It enforces memory safety without garbage collection.",
        }

        pkg = InternetLearnerEngine.learn(
            topic="Rust Language",
            project_root=self.project_root,
            deep_browse=True,
        )

        self.assertIsInstance(pkg, LearnedKnowledgePackage)
        self.assertEqual(pkg.topic, "Rust Language")
        self.assertGreater(len(pkg.extracted_facts), 0)
        self.assertGreater(pkg.hippocampus_facts_stored, 0)
        self.assertGreater(pkg.sft_samples_generated, 0)
        self.assertTrue(Path(pkg.saved_document_path).is_file())

        # Clean up created markdown test file
        try:
            Path(pkg.saved_document_path).unlink()
        except Exception:
            pass

    def test_format_learning_report(self):
        pkg = LearnedKnowledgePackage(
            topic="WASM",
            query="WASM",
            source_urls=["https://webassembly.org/"],
            extracted_facts=["WebAssembly is a binary instruction format."],
            summary="Synthesized knowledge about WASM.",
            saved_document_path="learned_wasm.md",
            hippocampus_facts_stored=1,
            sft_samples_generated=2,
            rag_indexed=True,
            elapsed_sec=1.23,
        )
        report = InternetLearnerEngine.format_learning_report(pkg)
        self.assertIn("WASM", report)
        self.assertIn("WebAssembly is a binary instruction format", report)
        self.assertIn("https://webassembly.org/", report)

    def test_tool_registry_learn_from_internet(self):
        reg = ToolRegistry()
        register_builtin_tools(reg, self.project_root)

        tool = reg.get("learn_from_internet")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "learn_from_internet")
        self.assertIn("topic", tool.parameters)

    def test_orchestrator_routes_internet_learning(self):
        orch = ReasoningOrchestrator(enable_thinking=True)
        intent = orch.classify_intent("Use DuckDuckGo and make it able to also learn from internet about Quantum Computing")
        self.assertEqual(intent, "internet-learning")


if __name__ == "__main__":
    unittest.main()
