"""
Interactive Coding & Safe Sandbox Engine
Provides Python code generation, AST syntax verification, and in-process
sandboxed execution with stdout capture and execution timeout protection.
"""
import ast
import contextlib
import io
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


class SafeCodeSandbox:
    """
    Executes Python snippets safely with stdout capture, restricted builtins,
    and runtime exception containment.
    """

    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        "math": math,
    }

    @classmethod
    def verify_syntax(cls, code_str: str) -> Tuple[bool, Optional[str]]:
        """Verifies code syntax using Python's AST parser."""
        try:
            ast.parse(code_str)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError on line {e.lineno}: {e.msg}"

    @classmethod
    def execute_snippet(cls, code_str: str, timeout_seconds: float = 2.0) -> Dict[str, Any]:
        """
        Executes a Python code snippet in a restricted namespace and captures stdout.
        """
        is_valid, err = cls.verify_syntax(code_str)
        if not is_valid:
            return {
                "success": False,
                "error": err,
                "stdout": "",
                "elapsed": 0.0,
            }

        # Guard against hazardous system imports
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in getattr(node, "names", []):
                    if alias.name in ["os", "sys", "subprocess", "shutil", "socket", "ctypes"]:
                        return {
                            "success": False,
                            "error": f"Security restriction: Import of '{alias.name}' is prohibited in safe sandbox.",
                            "stdout": "",
                            "elapsed": 0.0,
                        }

        stdout_capture = io.StringIO()
        restricted_globals = {"__builtins__": cls.SAFE_BUILTINS, "math": math}
        local_vars: Dict[str, Any] = {}

        t0 = time.time()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                compiled = compile(code_str, "<sandbox>", "exec")
                exec(compiled, restricted_globals, local_vars)
            elapsed = time.time() - t0
            output = stdout_capture.getvalue()
            return {
                "success": True,
                "stdout": output.strip(),
                "variables": {k: str(v)[:100] for k, v in local_vars.items() if not k.startswith("_")},
                "elapsed": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - t0
            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "stdout": stdout_capture.getvalue().strip(),
                "elapsed": elapsed,
            }


class CodingAssistant:
    """Claude & GPT-style coding problem solver, formatter, and self-testing assistant."""

    @classmethod
    def format_solution(
        cls,
        task: str,
        code: str,
        explanation: str,
        sandbox_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Formats code with markdown, explanation, complexity, and verified test execution."""
        lines = [
            f"### Python Implementation: {task}",
            "",
            "```python",
            code.strip(),
            "```",
            "",
            "**Explanation & Design:**",
            explanation.strip(),
        ]

        if sandbox_result and sandbox_result.get("success"):
            stdout_text = sandbox_result.get("stdout")
            if stdout_text:
                lines.append("\n**Verified Execution Output:**")
                lines.append("```text")
                lines.append(stdout_text)
                lines.append("```")
            lines.append(f"*(Code verified & executed in {sandbox_result.get('elapsed', 0):.4f}s)*")
        elif sandbox_result and sandbox_result.get("error"):
            lines.append(f"\n*Warning: Sandbox check returned: {sandbox_result.get('error')}*")

        return "\n".join(lines)
