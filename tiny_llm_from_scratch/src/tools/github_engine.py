"""
GitHub & Git Automation Engine
Provides comprehensive GitHub API, GitHub CLI (gh), and local Git integration:
- Repository discovery & search (topics, stars, languages)
- Remote code & file inspection from any GitHub repository
- Issue and Pull Request tracking & creation
- Local Git operations (status, diff, log, commit, branch)
- Autonomous Agent planning integration
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request


class GitHubEngine:
    """Robust GitHub and Git automation engine with 'gh' CLI and REST API fallback."""

    USER_AGENT = "Antigravity-GitHub-Agent/2.2"
    DEFAULT_TIMEOUT_SEC = 10

    @classmethod
    def _run_cmd(cls, args: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        try:
            res = subprocess.run(
                args,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=cls.DEFAULT_TIMEOUT_SEC,
                encoding="utf-8",
                errors="replace",
            )
            if res.returncode == 0:
                return True, res.stdout.strip()
            return False, (res.stderr.strip() or res.stdout.strip() or f"Process exited with code {res.returncode}")
        except subprocess.TimeoutExpired:
            return False, f"Command '{' '.join(args)}' timed out after {cls.DEFAULT_TIMEOUT_SEC}s"
        except Exception as e:
            return False, f"Command execution error: {str(e)}"

    # -------------------------------------------------------------------------
    # 1. GitHub Remote API & 'gh' CLI Operations
    # -------------------------------------------------------------------------

    @classmethod
    def search_repositories(cls, query: str, limit: int = 5) -> Dict[str, Any]:
        """Searches GitHub repositories by query, keyword, topic, or language."""
        # Try 'gh search repos' first
        if shutil.which("gh"):
            success, output = cls._run_cmd(["gh", "search", "repos", query, "--limit", str(limit), "--json", "fullName,description,stargazersCount,forksCount,url"])
            if success and output:
                try:
                    data = json.loads(output)
                    return {"success": True, "provider": "gh_cli", "query": query, "count": len(data), "repositories": data}
                except Exception:
                    pass

        # Fallback to GitHub REST API
        try:
            encoded_q = urllib.parse.quote(query)
            url = f"https://api.github.com/search/repositories?q={encoded_q}&per_page={limit}"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT, "Accept": "application/vnd.github.v3+json"})
            with urllib.request.urlopen(req, timeout=cls.DEFAULT_TIMEOUT_SEC) as resp:
                raw = json.loads(resp.read().decode("utf-8", errors="replace"))

            items = [
                {
                    "fullName": r.get("full_name"),
                    "description": r.get("description"),
                    "stargazersCount": r.get("stargazers_count"),
                    "forksCount": r.get("forks_count"),
                    "url": r.get("html_url"),
                    "language": r.get("language"),
                }
                for r in raw.get("items", [])[:limit]
            ]
            return {"success": True, "provider": "rest_api", "query": query, "count": len(items), "repositories": items}
        except Exception as e:
            return {"success": False, "query": query, "error": f"Search failed: {str(e)}"}

    @classmethod
    def get_repo_info(cls, repo: str) -> Dict[str, Any]:
        """Gets metadata for a specific repository (e.g. 'owner/repo')."""
        clean_repo = repo.strip().replace("https://github.com/", "").strip("/")
        if shutil.which("gh"):
            success, output = cls._run_cmd(["gh", "repo", "view", clean_repo, "--json", "name,owner,description,stargazerCount,forkCount,isPrivate,defaultBranchRef"])
            if success and output:
                try:
                    return {"success": True, "repo": clean_repo, "data": json.loads(output)}
                except Exception:
                    pass

        try:
            url = f"https://api.github.com/repos/{clean_repo}"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT, "Accept": "application/vnd.github.v3+json"})
            with urllib.request.urlopen(req, timeout=cls.DEFAULT_TIMEOUT_SEC) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return {
                "success": True,
                "repo": clean_repo,
                "data": {
                    "name": data.get("name"),
                    "owner": data.get("owner", {}).get("login"),
                    "description": data.get("description"),
                    "stargazerCount": data.get("stargazers_count"),
                    "forkCount": data.get("forks_count"),
                    "defaultBranch": data.get("default_branch"),
                    "url": data.get("html_url"),
                },
            }
        except Exception as e:
            return {"success": False, "repo": clean_repo, "error": f"Failed to retrieve repo: {str(e)}"}

    @classmethod
    def get_file_content(cls, repo: str, file_path: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """Fetches raw file content from a GitHub repository."""
        clean_repo = repo.strip().replace("https://github.com/", "").strip("/")
        clean_path = file_path.strip().lstrip("/")
        branch_name = branch or "main"

        # Try raw.githubusercontent.com first
        raw_url = f"https://raw.githubusercontent.com/{clean_repo}/{branch_name}/{clean_path}"
        try:
            req = urllib.request.Request(raw_url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=cls.DEFAULT_TIMEOUT_SEC) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            return {
                "success": True,
                "repo": clean_repo,
                "path": clean_path,
                "branch": branch_name,
                "length": len(content),
                "content": content,
            }
        except urllib.error.HTTPError as e:
            if e.code == 404 and branch_name == "main":
                # Try 'master' branch fallback
                master_url = f"https://raw.githubusercontent.com/{clean_repo}/master/{clean_path}"
                try:
                    req = urllib.request.Request(master_url, headers={"User-Agent": cls.USER_AGENT})
                    with urllib.request.urlopen(req, timeout=cls.DEFAULT_TIMEOUT_SEC) as resp:
                        content = resp.read().decode("utf-8", errors="replace")
                    return {
                        "success": True,
                        "repo": clean_repo,
                        "path": clean_path,
                        "branch": "master",
                        "length": len(content),
                        "content": content,
                    }
                except Exception:
                    pass
            return {"success": False, "repo": clean_repo, "path": clean_path, "error": f"HTTP Error {e.code}: {e.reason}"}
        except Exception as e:
            return {"success": False, "repo": clean_repo, "path": clean_path, "error": str(e)}

    @classmethod
    def list_issues(cls, repo: str, state: str = "open", limit: int = 5) -> Dict[str, Any]:
        """Lists repository issues."""
        clean_repo = repo.strip().replace("https://github.com/", "").strip("/")
        if shutil.which("gh"):
            success, output = cls._run_cmd(["gh", "issue", "list", "--repo", clean_repo, "--state", state, "--limit", str(limit), "--json", "number,title,state,url"])
            if success and output:
                try:
                    return {"success": True, "repo": clean_repo, "issues": json.loads(output)}
                except Exception:
                    pass

        try:
            url = f"https://api.github.com/repos/{clean_repo}/issues?state={state}&per_page={limit}"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT, "Accept": "application/vnd.github.v3+json"})
            with urllib.request.urlopen(req, timeout=cls.DEFAULT_TIMEOUT_SEC) as resp:
                raw = json.loads(resp.read().decode("utf-8", errors="replace"))
            issues = [
                {"number": it.get("number"), "title": it.get("title"), "state": it.get("state"), "url": it.get("html_url")}
                for it in raw[:limit] if "pull_request" not in it
            ]
            return {"success": True, "repo": clean_repo, "issues": issues}
        except Exception as e:
            return {"success": False, "repo": clean_repo, "error": str(e)}

    @classmethod
    def list_pull_requests(cls, repo: str, state: str = "open", limit: int = 5) -> Dict[str, Any]:
        """Lists repository pull requests."""
        clean_repo = repo.strip().replace("https://github.com/", "").strip("/")
        if shutil.which("gh"):
            success, output = cls._run_cmd(["gh", "pr", "list", "--repo", clean_repo, "--state", state, "--limit", str(limit), "--json", "number,title,state,url,author"])
            if success and output:
                try:
                    return {"success": True, "repo": clean_repo, "pull_requests": json.loads(output)}
                except Exception:
                    pass
        return {"success": False, "repo": clean_repo, "error": "gh CLI required for advanced PR queries."}

    # -------------------------------------------------------------------------
    # 2. Local Git Operations (Status, Diff, Log, Commit)
    # -------------------------------------------------------------------------

    @classmethod
    def git_status(cls, project_root: Optional[Path] = None) -> Dict[str, Any]:
        """Gets local git status (branch, modified files, untracked files)."""
        root = project_root or Path.cwd()
        success, output = cls._run_cmd(["git", "status", "--short", "--branch"], cwd=root)
        if not success:
            return {"success": False, "error": output}

        lines = output.splitlines()
        branch_line = lines[0] if lines else ""
        changed_files = lines[1:] if len(lines) > 1 else []

        return {
            "success": True,
            "branch": branch_line.lstrip("# ").strip(),
            "has_changes": len(changed_files) > 0,
            "changed_files_count": len(changed_files),
            "files": changed_files[:25],
        }

    @classmethod
    def git_diff(cls, project_root: Optional[Path] = None, max_lines: int = 150) -> Dict[str, Any]:
        """Gets uncommitted git diff."""
        root = project_root or Path.cwd()
        success, output = cls._run_cmd(["git", "diff", "--stat"], cwd=root)
        if not success:
            return {"success": False, "error": output}

        _, patch = cls._run_cmd(["git", "diff"], cwd=root)
        patch_lines = patch.splitlines()
        truncated = False
        if len(patch_lines) > max_lines:
            patch = "\n".join(patch_lines[:max_lines]) + f"\n\n[... Truncated: {len(patch_lines)} total diff lines ...]"
            truncated = True

        return {
            "success": True,
            "summary": output,
            "patch": patch,
            "truncated": truncated,
        }

    @classmethod
    def git_log(cls, project_root: Optional[Path] = None, limit: int = 5) -> Dict[str, Any]:
        """Gets recent commit history."""
        root = project_root or Path.cwd()
        success, output = cls._run_cmd(["git", "log", f"-{limit}", "--oneline", "--decorate"], cwd=root)
        if not success:
            return {"success": False, "error": output}

        commits = [{"commit": line.split()[0], "message": " ".join(line.split()[1:])} for line in output.splitlines() if line]
        return {
            "success": True,
            "count": len(commits),
            "commits": commits,
        }

    @classmethod
    def git_commit(cls, message: str, project_root: Optional[Path] = None) -> Dict[str, Any]:
        """Stages all tracked changes and creates a git commit."""
        root = project_root or Path.cwd()
        cls._run_cmd(["git", "add", "-u"], cwd=root)
        success, output = cls._run_cmd(["git", "commit", "-m", message], cwd=root)
        return {
            "success": success,
            "message": message,
            "output": output,
        }

    # -------------------------------------------------------------------------
    # 3. Formatting Helpers
    # -------------------------------------------------------------------------

    @classmethod
    def format_repos_display(cls, repos_result: Dict[str, Any]) -> str:
        """Formats repository search result into clean text card."""
        if not repos_result.get("success"):
            return f"[GITHUB ERROR] {repos_result.get('error')}"

        repos = repos_result.get("repositories", [])
        lines = [
            f"=== [GITHUB SEARCH RESULTS: '{repos_result.get('query')}'] ({len(repos)} repos) ===",
        ]
        for r in repos:
            stars = r.get("stargazersCount") or r.get("stargazerCount") or 0
            lines.append(f"- **{r.get('fullName', r.get('name'))}** (★ {stars:,})")
            if r.get("description"):
                lines.append(f"  {r.get('description')}")
            lines.append(f"  URL: {r.get('url')}")
        return "\n".join(lines)
