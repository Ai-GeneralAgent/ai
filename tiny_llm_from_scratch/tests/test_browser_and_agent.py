"""
Unit and Integration Tests for Web Browser Engine & Autonomous Work Agent
"""
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NESTED_ROOT = PROJECT_ROOT / "tiny_llm_from_scratch"

for p in (PROJECT_ROOT, NESTED_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.tools.browser import CleanHTMLTextExtractor, WebBrowserEngine
from src.agents.autonomous_agent import AutonomousWorkAgent, AgentMissionResult
from src.tools.registry import ToolRegistry, ToolDefinition
from src.tools.builtin import register_builtin_tools


class TestBrowserAndAutonomousAgent(unittest.TestCase):
    """Verifies headless browsing, HTML extraction, and autonomous agent execution."""

    def setUp(self):
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry, PROJECT_ROOT)

    def test_html_clean_text_extractor(self):
        """CleanHTMLTextExtractor must strip script/style tags and extract title, clean text, and links."""
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Python 3.14 Development Update</title>
            <style>body { color: red; } .nav { display: none; }</style>
            <script>alert("malicious script should be stripped");</script>
        </head>
        <body>
            <h1>Python 3.14 Pre-Release</h1>
            <p>Python 3.14 introduces free-threaded CPython improvements and JIT compiler enhancements.</p>
            <div>For more details, visit <a href="https://docs.python.org/3.14/">Python Documentation</a>.</div>
        </body>
        </html>
        """
        extractor = CleanHTMLTextExtractor(base_url="https://python.org")
        extractor.feed(sample_html)
        clean_text = extractor.get_clean_text()

        self.assertEqual(extractor.title, "Python 3.14 Development Update")
        self.assertIn("Python 3.14 Pre-Release", clean_text)
        self.assertIn("free-threaded CPython", clean_text)
        self.assertNotIn("alert(", clean_text)
        self.assertNotIn("body { color: red; }", clean_text)

        self.assertEqual(len(extractor.links), 1)
        self.assertEqual(extractor.links[0]["text"], "Python Documentation")
        self.assertEqual(extractor.links[0]["url"], "https://docs.python.org/3.14/")

    def test_browser_tools_registered(self):
        """Browser tools must be registered and available in the ToolRegistry."""
        browse_tool = self.registry.get_tool("browse_web_page")
        self.assertIsNotNone(browse_tool)
        self.assertIn("url", browse_tool.parameters)

        search_browse = self.registry.get_tool("search_and_browse")
        self.assertIsNotNone(search_browse)

        agent_tool = self.registry.get_tool("run_autonomous_agent")
        self.assertIsNotNone(agent_tool)

    def test_autonomous_agent_plan_formulation(self):
        """AutonomousWorkAgent must formulate structured plans matching user goals."""
        agent = AutonomousWorkAgent(self.registry, max_steps=5, verbose=False)

        # Plan for URL goal
        url_plan = agent.formulate_mission_plan("Browse https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)")
        self.assertGreater(len(url_plan), 0)
        self.assertEqual(url_plan[0]["tool"], "browse_web_page")

        # Plan for Search goal
        search_plan = agent.formulate_mission_plan("Research the latest news on quantum computing")
        self.assertGreater(len(search_plan), 0)
        self.assertIn("search", search_plan[0]["tool"])

        # Plan for Security Audit goal
        sec_plan = agent.formulate_mission_plan("Audit code and check security vulnerabilities")
        self.assertGreater(len(sec_plan), 0)
        self.assertIn("audit", sec_plan[1]["tool"])

    def test_autonomous_agent_mission_execution(self):
        """Agent must execute multi-step mission, collect evidence, and synthesize results."""
        agent = AutonomousWorkAgent(self.registry, max_steps=4, timeout_sec=15.0, verbose=False)
        result = agent.execute_mission("Audit code and check security vulnerabilities")

        self.assertTrue(result.completed)
        self.assertGreater(result.iterations, 0)
        self.assertIsInstance(result.synthesized_output, str)
        self.assertIn("Autonomous Agent", result.synthesized_output)


if __name__ == "__main__":
    unittest.main()

