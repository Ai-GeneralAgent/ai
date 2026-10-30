"""
Advanced Software Development Tools
Provides:
- Multi-file project analysis and structural overview
- Python AST code understanding (classes, functions, call hierarchy, complexity)
- Dependency inspection (requirements.txt, package versions)
- Bug localization and root-cause analysis
- Test scaffolding and code review
"""
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CodeSymbol:
    name: str
    kind: str  # function, class, async_function, method
    line_start: int
    line_end: int
    docstring: Optional[str]
    parameters: List[str]
    complexity_estimate: int


@dataclass
class FileAnalysis:
    file_path: str
    line_count: int
    symbols: List[CodeSymbol]
    imports: List[str]
    classes: List[str]
    functions: List[str]


class ProjectCodeAnalyzer:
    """Multi-file code understanding, architecture, and static analysis."""

    @classmethod
    def analyze_file(cls, file_path: Path) -> FileAnalysis:
        p = Path(file_path)
        if not p.is_file():
            return FileAnalysis(str(p), 0, [], [], [], [])

        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            tree = ast.parse(content, filename=p.name)
        except Exception:
            return FileAnalysis(str(p), 0, [], [], [], [])

        imports: List[str] = []
        symbols: List[CodeSymbol] = []
        classes: List[str] = []
        functions: List[str] = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    imports.append(f"{mod}.{alias.name}")

            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                doc = ast.get_docstring(node)
                end = getattr(node, "end_lineno", node.lineno)
                symbols.append(CodeSymbol(
                    name=node.name,
                    kind="class",
                    line_start=node.lineno,
                    line_end=end,
                    docstring=doc,
                    parameters=[b.id for b in node.bases if isinstance(b, ast.Name)],
                    complexity_estimate=len(node.body),
                ))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        f_doc = ast.get_docstring(item)
                        f_end = getattr(item, "end_lineno", item.lineno)
                        params = [a.arg for a in item.args.args]
                        symbols.append(CodeSymbol(
                            name=f"{node.name}.{item.name}",
                            kind="method",
                            line_start=item.lineno,
                            line_end=f_end,
                            docstring=f_doc,
                            parameters=params,
                            complexity_estimate=cls._cyclomatic_complexity(item),
                        ))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
                doc = ast.get_docstring(node)
                end = getattr(node, "end_lineno", node.lineno)
                params = [a.arg for a in node.args.args]
                symbols.append(CodeSymbol(
                    name=node.name,
                    kind="function",
                    line_start=node.lineno,
                    line_end=end,
                    docstring=doc,
                    parameters=params,
                    complexity_estimate=cls._cyclomatic_complexity(node),
                ))

        return FileAnalysis(
            file_path=str(p),
            line_count=len(lines),
            symbols=symbols,
            imports=imports,
            classes=classes,
            functions=functions,
        )

    @classmethod
    def analyze_project_structure(cls, root_dir: Path, max_depth: int = 4) -> Dict[str, Any]:
        """Scans project tree and builds structural architecture map."""
        root = Path(root_dir)
        py_files: List[Path] = []
        for p in root.rglob("*.py"):
            parts = p.parts
            if any(ign in parts for ign in [".git", ".venv", "__pycache__", "build", "dist"]):
                continue
            py_files.append(p)

        total_lines = 0
        all_classes = 0
        all_functions = 0
        file_summaries = []

        for pf in sorted(py_files)[:60]:
            analysis = cls.analyze_file(pf)
            total_lines += analysis.line_count
            all_classes += len(analysis.classes)
            all_functions += len(analysis.functions)
            file_summaries.append({
                "path": str(pf.relative_to(root)),
                "lines": analysis.line_count,
                "classes": analysis.classes,
                "functions": analysis.functions,
            })

        return {
            "root": str(root),
            "python_files_count": len(py_files),
            "total_lines_of_code": total_lines,
            "total_classes": all_classes,
            "total_functions": all_functions,
            "files": file_summaries,
        }

    @classmethod
    def inspect_dependencies(cls, requirements_path: Path) -> Dict[str, Any]:
        """Parses requirements.txt, checking dependencies and version specifications."""
        req_file = Path(requirements_path)
        if not req_file.is_file():
            return {"error": f"Requirements file not found: {req_file}"}

        lines = req_file.read_text(encoding="utf-8").splitlines()
        deps = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = re.split(r"(==|>=|<=|~=|>|<)", stripped)
            name = parts[0].strip()
            spec = "".join(parts[1:]).strip() if len(parts) > 1 else "unpinned"
            deps.append({"package": name, "specification": spec})

        return {
            "file": str(req_file),
            "dependency_count": len(deps),
            "dependencies": deps,
        }

    @staticmethod
    def _cyclomatic_complexity(node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
