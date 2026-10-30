"""
Unit tests for GitHubEngine and Git Automation Tools
Covers remote repository querying, file fetching, issue tracking,
local git status/diff/log, formatting, and tool registry integration.
"""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from src.tools.github_engine import GitHubEngine
from src.tools.registry import ToolRegistry
from src.tools.builtin import register_builtin_tools


class TestGitHubEngine(unittest.TestCase):
    """Test suite for GitHubEngine functions and CLI/API fallbacks."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent

    @patch("subprocess.run")
    def test_search_repositories_via_gh(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"fullName": "torvalds/linux", "description": "Linux kernel source tree", "stargazersCount": 185000, "forksCount": 54000, "url": "https://github.com/torvalds/linux"}
            ]),
            stderr=""
        )
        res = GitHubEngine.search_repositories("linux", limit=1)
        self.assertTrue(res["success"])
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["repositories"][0]["fullName"], "torvalds/linux")

    @patch("urllib.request.urlopen")
    @patch("shutil.which", return_value=None)
    def test_search_repositories_via_rest(self, mock_which, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "items": [
                {"full_name": "python/cpython", "description": "The Python programming language", "stargazers_count": 65000, "forks_count": 30000, "html_url": "https://github.com/python/cpython", "language": "Python"}
            ]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        res = GitHubEngine.search_repositories("python", limit=1)
        self.assertTrue(res["success"])
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["repositories"][0]["fullName"], "python/cpython")

    @patch("subprocess.run")
    def test_get_repo_info_via_gh(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "name": "cpython",
                "owner": {"login": "python"},
                "description": "The Python programming language",
                "stargazerCount": 65000,
                "forkCount": 30000,
                "isPrivate": False,
                "defaultBranchRef": {"name": "main"}
            }),
            stderr=""
        )
        res = GitHubEngine.get_repo_info("python/cpython")
        self.assertTrue(res["success"])
        self.assertEqual(res["data"]["name"], "cpython")

    @patch("urllib.request.urlopen")
    def test_get_file_content_success(self, mock_urlopen):
        sample_code = "print('Hello from GitHub!')\n"
        mock_response = MagicMock()
        mock_response.read.return_value = sample_code.encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        res = GitHubEngine.get_file_content("owner/repo", "main.py")
        self.assertTrue(res["success"])
        self.assertEqual(res["content"], sample_code)
        self.assertEqual(res["length"], len(sample_code))

    @patch("urllib.request.urlopen")
    def test_get_file_content_master_fallback(self, mock_urlopen):
        sample_code = "def foo(): pass"
        error_404 = urllib.error.HTTPError("https://...", 404, "Not Found", {}, None)
        success_resp = MagicMock()
        success_resp.read.return_value = sample_code.encode("utf-8")
        success_resp.__enter__.return_value = success_resp

        mock_urlopen.side_effect = [error_404, success_resp]

        res = GitHubEngine.get_file_content("owner/repo", "setup.py")
        self.assertTrue(res["success"])
        self.assertEqual(res["branch"], "master")
        self.assertEqual(res["content"], sample_code)

    @patch("subprocess.run")
    def test_list_issues_via_gh(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"number": 101, "title": "Add async support", "state": "OPEN", "url": "https://github.com/foo/bar/issues/101"}
            ]),
            stderr=""
        )
        res = GitHubEngine.list_issues("foo/bar", limit=1)
        self.assertTrue(res["success"])
        self.assertEqual(len(res["issues"]), 1)
        self.assertEqual(res["issues"][0]["number"], 101)

    def test_local_git_status_live(self):
        res = GitHubEngine.git_status(self.project_root)
        self.assertTrue(res["success"])
        self.assertIn("branch", res)
        self.assertIsInstance(res["files"], list)

    def test_local_git_diff_live(self):
        res = GitHubEngine.git_diff(self.project_root, max_lines=50)
        self.assertTrue(res["success"])
        self.assertIn("summary", res)
        self.assertIn("patch", res)

    def test_local_git_log_live(self):
        res = GitHubEngine.git_log(self.project_root, limit=3)
        self.assertTrue(res["success"])
        self.assertGreater(res["count"], 0)
        self.assertIn("commit", res["commits"][0])
        self.assertIn("message", res["commits"][0])

    def test_format_repos_display(self):
        repos_data = {
            "success": True,
            "query": "transformer",
            "repositories": [
                {"fullName": "huggingface/transformers", "stargazersCount": 135000, "description": "State-of-the-art Machine Learning", "url": "https://github.com/huggingface/transformers"}
            ]
        }
        card = GitHubEngine.format_repos_display(repos_data)
        self.assertIn("huggingface/transformers", card)
        self.assertIn("135,000", card)

    def test_tool_registry_github_integration(self):
        reg = ToolRegistry()
        register_builtin_tools(reg, self.project_root)

        expected_tools = [
            "github_search_repos",
            "github_get_repo",
            "github_view_file",
            "github_list_issues",
            "github_list_prs",
            "git_local_status",
            "git_local_diff",
            "git_local_log",
            "git_local_commit",
        ]
        for t_name in expected_tools:
            tool = reg.get(t_name)
            self.assertIsNotNone(tool, f"Missing tool: {t_name}")

        # Execute git_local_status through ToolRegistry
        result = reg.execute("git_local_status", {})
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
