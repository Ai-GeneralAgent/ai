import ast
from datetime import datetime
import math
import operator
from pathlib import Path
from typing import Any, Dict, List, Union

from .registry import ToolDefinition, ToolRegistry


class SafeCalculator:
    """Evaluates mathematical expressions safely using an AST whitelist (no eval)."""

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    FUNCTIONS = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "round": round,
    }

    @classmethod
    def evaluate(cls, expression: str) -> Union[int, float]:
        node = ast.parse(expression.strip(), mode="eval").body
        return cls._eval_node(node)

    @classmethod
    def _eval_node(cls, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            return cls.OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = cls._eval_node(node.operand)
            return cls.OPERATORS[op_type](operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in cls.FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            args = [cls._eval_node(arg) for arg in node.args]
            return cls.FUNCTIONS[func_name](*args)
        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")


def calculate_expression(expression: str) -> str:
    """Safe calculator tool function."""
    try:
        val = SafeCalculator.evaluate(expression)
        return str(val)
    except Exception as e:
        return f"Math error: {str(e)}"


def get_current_datetime() -> str:
    """Current date and time tool."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_project_file(relative_path: str, base_dir: Path = Path(".")) -> str:
    """Reads a permitted file within the project boundary with path traversal prevention."""
    base = base_dir.resolve()
    target = (base / relative_path).resolve()

    if not str(target).startswith(str(base)):
        raise PermissionError(f"Access denied: path traversal attempt outside project directory.")

    if not target.is_file():
        raise FileNotFoundError(f"File not found: {relative_path}")

    # Prevent reading virtual environments or large binary files
    if ".venv" in target.parts or ".git" in target.parts:
        raise PermissionError("Access denied to virtual environment or git internals.")

    if target.stat().st_size > 1_000_000:
        return f"File too large ({target.stat().st_size:,} bytes). Reading first 500 lines only:\n" + "\n".join(
            target.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:500]
        )

    return target.read_text(encoding="utf-8-sig", errors="replace")


def search_codebase(query: str, base_dir: Path = Path(".")) -> List[Dict[str, Any]]:
    """Searches for terms across project source files (.py, .json, .md)."""
    results = []
    base = base_dir.resolve()

    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".py", ".json", ".md"]:
            if ".venv" in p.parts or ".git" in p.parts or "__pycache__" in p.parts:
                continue
            try:
                lines = p.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                for idx, line in enumerate(lines, 1):
                    if query.lower() in line.lower():
                        results.append({
                            "file": str(p.relative_to(base)),
                            "line": idx,
                            "content": line.strip()[:150],
                        })
                        if len(results) >= 20:
                            return results
            except Exception:
                continue
    return results


def register_builtin_tools(registry: ToolRegistry, project_root: Path = Path(".")):
    """Registers standard safe tools into the tool registry."""
    registry.register(ToolDefinition(
        name="calculator",
        description="Safely evaluates arithmetic expressions (e.g., '14 * 12 + sqrt(16)').",
        parameters={"expression": "string arithmetic expression"},
        execute_fn=calculate_expression,
    ))

    registry.register(ToolDefinition(
        name="datetime_now",
        description="Returns current local date and time.",
        parameters={},
        execute_fn=get_current_datetime,
    ))

    registry.register(ToolDefinition(
        name="read_project_file",
        description="Reads permitted project text or code file.",
        parameters={"relative_path": "relative path to file"},
        execute_fn=lambda relative_path: read_project_file(relative_path, project_root),
    ))

    registry.register(ToolDefinition(
        name="search_codebase",
        description="Searches code and text files for a query string.",
        parameters={"query": "search query"},
        execute_fn=lambda query: search_codebase(query, project_root),
    ))

