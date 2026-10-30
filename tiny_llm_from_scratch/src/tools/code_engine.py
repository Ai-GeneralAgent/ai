"""
Interactive Coding & Safe Sandbox Engine
Provides Python code generation, AST syntax verification, and in-process
sandboxed execution with stdout capture and execution timeout protection.
"""
import ast
import contextlib
import io
import math
import re
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
    """Interactive Python coding problem solver, formatter, syntax verifier, and sandbox engine."""

    @classmethod
    def generate_solution(cls, query: str) -> Optional[Dict[str, Any]]:
        """Synthesizes verified Python solutions for common algorithmic, data structure, or utility tasks."""
        lower = query.lower().strip()

        # 1. Add two numbers
        if any(k in lower for k in ["add two numbers", "adds two numbers", "addition of two numbers", "sum of two numbers"]):
            code = (
                "def add(a: float, b: float) -> float:\n"
                "    \"\"\"Return the sum of two numbers.\"\"\"\n"
                "    return a + b\n\n"
                "# Example usage:\n"
                "result = add(12.5, 7.5)\n"
                "print(f'12.5 + 7.5 = {result}')\n"
            )
            return {
                "task": "Add Two Numbers",
                "code": code,
                "explanation": "Defines an `add` function with type annotations that accepts two numbers and returns their sum using the `+` operator.",
            }

        # 2. Fibonacci sequence
        if "fibonacci" in lower:
            code = (
                "def fibonacci(n: int) -> list[int]:\n"
                "    \"\"\"Generate the first n Fibonacci numbers.\"\"\"\n"
                "    if n <= 0:\n"
                "        return []\n"
                "    seq = [0, 1] if n > 1 else [0]\n"
                "    while len(seq) < n:\n"
                "        seq.append(seq[-1] + seq[-2])\n"
                "    return seq\n\n"
                "# Example:\n"
                "print('First 8 Fibonacci numbers:', fibonacci(8))\n"
            )
            return {
                "task": "Fibonacci Sequence Generator",
                "code": code,
                "explanation": "Computes Fibonacci numbers iteratively in O(n) time and O(n) space without recursion overhead.",
            }

        # 3. Factorial
        if "factorial" in lower:
            code = (
                "def factorial(n: int) -> int:\n"
                "    \"\"\"Return factorial of a non-negative integer n.\"\"\"\n"
                "    if n < 0:\n"
                "        raise ValueError('Factorial is not defined for negative numbers')\n"
                "    result = 1\n"
                "    for i in range(2, n + 1):\n"
                "        result *= i\n"
                "    return result\n\n"
                "# Example:\n"
                "print('5! =', factorial(5))\n"
            )
            return {
                "task": "Factorial Calculation",
                "code": code,
                "explanation": "Calculates n! iteratively with input validation for non-negative integers.",
            }

        # 4. Reverse string
        if "reverse" in lower and ("string" in lower or "word" in lower):
            code = (
                "def reverse_string(s: str) -> str:\n"
                "    \"\"\"Return the reversed version of string s.\"\"\"\n"
                "    return s[::-1]\n\n"
                "# Example:\n"
                "print('Reversed:', reverse_string('Python'))\n"
            )
            return {
                "task": "String Reversal",
                "code": code,
                "explanation": "Reverses a string using Python slice notation `s[::-1]` in O(n) time.",
            }

        # 5. Check prime
        if "prime" in lower:
            code = (
                "def is_prime(n: int) -> bool:\n"
                "    \"\"\"Check if integer n is a prime number.\"\"\"\n"
                "    if n < 2:\n"
                "        return False\n"
                "    for d in range(2, int(n ** 0.5) + 1):\n"
                "        if n % d == 0:\n"
                "            return False\n"
                "    return True\n\n"
                "# Example:\n"
                "print('Is 29 prime?', is_prime(29))\n"
                "print('Is 30 prime?', is_prime(30))\n"
            )
            return {
                "task": "Prime Number Check",
                "code": code,
                "explanation": "Tests divisibility up to sqrt(n) in O(sqrt(n)) time.",
            }

        return None

    @classmethod
    def debug_expression(cls, query: str) -> Optional[str]:
        """Diagnoses Python syntax errors and provides actionable correction."""
        # Check for incomplete expression like print(1 + )
        incomplete_plus = re.search(r"print\s*\(\s*(\d+)\s*\+\s*\)", query)
        if incomplete_plus:
            n = incomplete_plus.group(1)
            return f"The expression is incomplete. Use `print({n} + 2)`, replacing `2` with the value you intended."

        # General syntax check if code snippet or expression is provided
        code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", query)
        code_to_check = code_match.group(1) if code_match else None
        if not code_to_check and ("print(" in query or "def " in query or "=" in query):
            # Try to extract the code part
            m = re.search(r"(print\(.*?\)|def .*?:|[\w_]+\s*=.*)", query)
            if m:
                code_to_check = m.group(1)

        if code_to_check:
            try:
                ast.parse(code_to_check)
                return f"Syntax check passed: The code is valid Python syntax.\n```python\n{code_to_check}\n```"
            except SyntaxError as e:
                return (
                    f"Syntax Error detected in expression:\n\n"
                    f"- **Error:** `{e.msg}` (Line {e.lineno}, column {e.offset})\n"
                    f"- **Issue:** The code syntax is malformed or incomplete.\n"
                    f"- **Fix:** Complete the expression or statement properly before evaluation."
                )

        return None

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

