"""
Built-in Safe Tools Framework
Includes Advanced Math Solver (algebra, arithmetic, percentages, unit conversions, statistics),
Path-guarded File Reader, Codebase Search, and System DateTime.
"""
import ast
from datetime import datetime
import math
import operator
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

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

    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
    }

    FUNCTIONS = {
        "sqrt": math.sqrt,
        "cbrt": lambda x: math.pow(x, 1 / 3) if x >= 0 else -math.pow(-x, 1 / 3),
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "exp": math.exp,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "lcm": getattr(math, "lcm", lambda a, b: abs(a * b) // math.gcd(a, b)),
        "comb": getattr(math, "comb", lambda n, k: math.factorial(n) // (math.factorial(k) * math.factorial(n - k))),
        "perm": getattr(math, "perm", lambda n, k: math.factorial(n) // math.factorial(n - k)),
        "abs": abs,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
        "min": min,
        "max": max,
        "sum": sum,
    }

    @classmethod
    def evaluate(cls, expression: str) -> Union[int, float]:
        clean_expr = expression.strip().replace("^", "**")
        node = ast.parse(clean_expr, mode="eval").body
        return cls._eval_node(node)

    @classmethod
    def _eval_node(cls, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in cls.CONSTANTS:
                return cls.CONSTANTS[node.id]
            raise ValueError(f"Unknown variable or constant: {node.id}")
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
        elif isinstance(node, ast.List):
            return [cls._eval_node(elt) for elt in node.elts]
        raise ValueError(f"Unsupported expression construct: {type(node).__name__}")


class AdvancedMathSolver:
    """Solves algebraic equations, percentages, statistics, unit conversions, and word problems."""

    UNIT_CONVERSIONS = {
        # Length (base: meter)
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v / 0.621371,
        ("m", "feet"): lambda v: v * 3.28084,
        ("feet", "m"): lambda v: v / 3.28084,
        ("cm", "inches"): lambda v: v / 2.54,
        ("inches", "cm"): lambda v: v * 2.54,
        # Temperature
        ("c", "f"): lambda v: (v * 9 / 5) + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("c", "k"): lambda v: v + 273.15,
        ("k", "c"): lambda v: v - 273.15,
        # Weight (base: kg)
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v / 2.20462,
        ("g", "oz"): lambda v: v * 0.035274,
        ("oz", "g"): lambda v: v / 0.035274,
        # Speed
        ("km/h", "mph"): lambda v: v * 0.621371,
        ("mph", "km/h"): lambda v: v / 0.621371,
        ("m/s", "km/h"): lambda v: v * 3.6,
        ("km/h", "m/s"): lambda v: v / 3.6,
        # Digital
        ("gb", "mb"): lambda v: v * 1024,
        ("mb", "gb"): lambda v: v / 1024,
        ("tb", "gb"): lambda v: v * 1024,
        ("gb", "tb"): lambda v: v / 1024,
    }

    @classmethod
    def solve_linear_equation(cls, eq: str) -> Optional[Dict[str, Any]]:
        """Solves linear equations like ax + b = c or ax = c."""
        # Normalize
        eq = eq.replace(" ", "")
        if "=" not in eq:
            return None

        # Pattern: (a)?x (+/- b)? = c
        match = re.match(r"^([+-]?\d*(?:\.\d+)?)?\*?x([+-]\d+(?:\.\d+)?)?=([+-]?\d+(?:\.\d+)?)$", eq)
        if not match:
            return None

        a_str, b_str, c_str = match.groups()
        if a_str in ["", "+", None]:
            a = 1.0
        elif a_str == "-":
            a = -1.0
        else:
            a = float(a_str)

        b = float(b_str) if b_str else 0.0
        c = float(c_str)

        if a == 0:
            return None

        # ax + b = c => ax = c - b => x = (c - b) / a
        rhs = c - b
        x = rhs / a
        steps = [
            f"Original equation: {a}x + ({b}) = {c}",
            f"Subtract {b} from both sides: {a}x = {rhs}",
            f"Divide by {a}: x = {x}",
        ]
        return {"type": "linear_equation", "variable": "x", "result": x, "steps": steps}

    @classmethod
    def solve_quadratic_equation(cls, eq: str) -> Optional[Dict[str, Any]]:
        """Solves quadratic equations like ax^2 + bx + c = 0."""
        clean = eq.replace(" ", "").replace("**2", "^2")
        clean = re.sub(r"x2(?![0-9])", "x^2", clean)
        match = re.match(r"^([+-]?\d*(?:\.\d+)?)?\*?x\^2([+-]\d*(?:\.\d+)?\*?x)?([+-]\d+(?:\.\d+)?)?=0$", clean)
        if not match:
            return None

        a_str, b_str, c_str = match.groups()
        if a_str in ["", "+", None]:
            a = 1.0
        elif a_str == "-":
            a = -1.0
        else:
            a = float(a_str)

        if not b_str:
            b = 0.0
        else:
            b_val = b_str.replace("x", "").replace("*", "")
            b = 1.0 if b_val in ["+", ""] else (-1.0 if b_val == "-" else float(b_val))

        c = float(c_str) if c_str else 0.0

        discriminant = (b ** 2) - (4 * a * c)
        if discriminant > 0:
            x1 = (-b + math.sqrt(discriminant)) / (2 * a)
            x2 = (-b - math.sqrt(discriminant)) / (2 * a)
            roots = [round(x1, 4), round(x2, 4)]
        elif discriminant == 0:
            x1 = -b / (2 * a)
            roots = [round(x1, 4)]
        else:
            real = -b / (2 * a)
            imag = math.sqrt(-discriminant) / (2 * a)
            roots = [f"{real:.3f} + {imag:.3f}i", f"{real:.3f} - {imag:.3f}i"]

        steps = [
            f"Quadratic coefficients: a = {a}, b = {b}, c = {c}",
            f"Discriminant Delta = b^2 - 4ac = ({b})^2 - 4({a})({c}) = {discriminant}",
            f"Roots: {roots}",
        ]
        return {"type": "quadratic_equation", "result": roots, "steps": steps}

    @classmethod
    def solve_percentage(cls, query: str) -> Optional[Dict[str, Any]]:
        """Handles percentage queries like '20% of 150' or 'increase 100 by 15%'."""
        lower = query.lower()
        # Pattern: X% of Y
        m1 = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", lower)
        if m1:
            pct, total = float(m1.group(1)), float(m1.group(2))
            val = (pct / 100.0) * total
            return {
                "type": "percentage",
                "result": val,
                "steps": [f"Calculate {pct}% of {total}: ({pct} / 100) * {total} = {val}"],
            }

        # Pattern: increase/decrease X by Y%
        m2 = re.search(r"(increase|decrease|discount|off)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%\s*(?:on|from)?\s*(\d+(?:\.\d+)?)", lower)
        if m2:
            action, pct, base = m2.group(1), float(m2.group(2)), float(m2.group(3))
            delta = (pct / 100.0) * base
            val = (base + delta) if action == "increase" else (base - delta)
            return {
                "type": "percentage_delta",
                "result": val,
                "steps": [
                    f"Base: {base}, Percentage: {pct}%",
                    f"Change amount: ({pct} / 100) * {base} = {delta}",
                    f"Final result: {base} {'+' if action == 'increase' else '-'} {delta} = {val}",
                ],
            }
        return None

    @classmethod
    def convert_units(cls, query: str) -> Optional[Dict[str, Any]]:
        """Handles unit conversions like 'convert 100 km to miles' or '50 c to f'."""
        pattern = r"(?:convert\s+)?(\d+(?:\.\d+)?)\s*([a-zA-Z°/]+)\s+(?:to|in|into)\s+([a-zA-Z°/]+)"
        match = re.search(pattern, query.lower())
        if not match:
            return None

        val, from_u, to_u = float(match.group(1)), match.group(2).replace("°", "").strip(), match.group(3).replace("°", "").strip()

        # Normalize common names
        norm_map = {
            "kilometer": "km", "kilometers": "km", "mile": "miles", "metre": "m", "meter": "m",
            "meters": "m", "foot": "feet", "inch": "inches", "celsius": "c", "centigrade": "c",
            "fahrenheit": "f", "kelvin": "k", "kilogram": "kg", "kilograms": "kg", "pound": "lbs",
            "pounds": "lbs", "gram": "g", "grams": "g", "ounce": "oz", "ounces": "oz",
        }
        from_norm = norm_map.get(from_u, from_u)
        to_norm = norm_map.get(to_u, to_u)

        key = (from_norm, to_norm)
        if key in cls.UNIT_CONVERSIONS:
            converted = cls.UNIT_CONVERSIONS[key](val)
            steps = [f"Convert {val} {from_u} to {to_u}", f"Result: {converted:.4f} {to_u}"]
            return {"type": "unit_conversion", "result": round(converted, 4), "steps": steps}

        return None

    @classmethod
    def solve_statistics(cls, numbers: List[float]) -> Dict[str, Any]:
        """Calculates standard descriptive statistics."""
        if not numbers:
            raise ValueError("Number list cannot be empty.")
        n = len(numbers)
        s_mean = sum(numbers) / n
        s_sorted = sorted(numbers)
        s_median = s_sorted[n // 2] if n % 2 != 0 else (s_sorted[n // 2 - 1] + s_sorted[n // 2]) / 2
        variance = sum((x - s_mean) ** 2 for x in numbers) / max(1, n - 1)
        stdev = math.sqrt(variance)

        return {
            "count": n,
            "sum": round(sum(numbers), 4),
            "mean": round(s_mean, 4),
            "median": round(s_median, 4),
            "min": min(numbers),
            "max": max(numbers),
            "stdev": round(stdev, 4),
        }


def solve_math_problem(query: str) -> Dict[str, Any]:
    """Unified entry-point for all mathematical and quantitative problems."""
    raw = query.strip().rstrip("?!. \t")
    clean = re.sub(r"(?i)^(solve|calculate|what is|compute|find|evaluate)\s*:?\s*", "", raw).strip("?!. \t")

    # Support common English phrasing
    clean = re.sub(r"(?i)square root of\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", clean)
    clean = re.sub(r"(?i)cube root of\s*(\d+(?:\.\d+)?)", r"cbrt(\1)", clean)
    clean = re.sub(r"(?i)\bmultiplied by\b", "*", clean)
    clean = re.sub(r"(?i)\btimes\b", "*", clean)
    clean = re.sub(r"(?i)\bdivided by\b", "/", clean)
    clean = re.sub(r"(?i)\bplus\b", "+", clean)
    clean = re.sub(r"(?i)\bminus\b", "-", clean)

    candidates = [clean, raw]

    # Extract any arithmetic / algebraic substring
    math_match = re.search(r"(\d+[\d\s\+\-\*\/\^\.\(\)]+[\d\)])", clean)
    if math_match:
        candidates.append(math_match.group(1).strip())

    for target in candidates:
        if not target:
            continue

        # 1. Algebraic Linear Equations
        lin_res = AdvancedMathSolver.solve_linear_equation(target)
        if lin_res:
            steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(lin_res["steps"]))
            return {
                "success": True,
                "result": lin_res["result"],
                "explanation": f"**Algebraic Solution:**\n{steps_text}\n\n**Final Answer:** `x = {lin_res['result']}`",
            }

        # 2. Quadratic Equations
        quad_res = AdvancedMathSolver.solve_quadratic_equation(target)
        if quad_res:
            steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(quad_res["steps"]))
            return {
                "success": True,
                "result": quad_res["result"],
                "explanation": f"**Quadratic Solution:**\n{steps_text}\n\n**Roots:** `{quad_res['result']}`",
            }

        # 3. Unit Conversions
        conv_res = AdvancedMathSolver.convert_units(target)
        if conv_res:
            return {
                "success": True,
                "result": conv_res["result"],
                "explanation": f"**Unit Conversion:**\n  • {conv_res['steps'][0]}\n\n**Result:** **{conv_res['result']}**",
            }

        # 4. Percentages
        pct_res = AdvancedMathSolver.solve_percentage(target)
        if pct_res:
            steps_text = "\n".join(f"  • {s}" for s in pct_res["steps"])
            return {
                "success": True,
                "result": pct_res["result"],
                "explanation": f"**Percentage Calculation:**\n{steps_text}\n\n**Final Answer:** **{pct_res['result']}**",
            }

    # 5. Direct Scientific / Arithmetic Evaluation
    for target in candidates:
        if not target:
            continue
        try:
            val = SafeCalculator.evaluate(target)
            return {
                "success": True,
                "result": val,
                "explanation": f"**Calculation:** `{target}` = **{val}**",
            }
        except Exception:
            pass

    return {"success": False, "error": f"Could not solve: {query}"}


def calculate_expression(expression: str) -> str:
    """Safe calculator tool function with unified math solver fallback."""
    res = solve_math_problem(expression)
    if res.get("success"):
        return str(res["result"])
    return f"Math error: {res.get('error', 'Unable to evaluate')}"


def get_current_datetime() -> str:
    """Current date and time tool."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_project_file(relative_path: str, base_dir: Path = Path(".")) -> str:
    """Reads a permitted file within the project boundary with path traversal prevention."""
    base = base_dir.resolve()
    target = (base / relative_path).resolve()

    if not str(target).startswith(str(base)):
        raise PermissionError("Access denied: path traversal attempt outside project directory.")

    if not target.is_file():
        raise FileNotFoundError(f"File not found: {relative_path}")

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
        description="Evaluates arithmetic and scientific expressions (e.g., '14 * 12 + sqrt(16)').",
        parameters={"expression": "string arithmetic or scientific expression"},
        execute_fn=calculate_expression,
    ))

    registry.register(ToolDefinition(
        name="math_solver",
        description="Solves algebraic equations (ax+b=c, ax^2+bx+c=0), percentages, and unit conversions.",
        parameters={"query": "mathematical query, equation, or conversion string"},
        execute_fn=lambda query: solve_math_problem(query).get("explanation", "Could not solve"),
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

    registry.register(ToolDefinition(
        name="python_sandbox",
        description="Safely executes a Python code snippet and captures stdout output.",
        parameters={"code": "valid python code string"},
        execute_fn=lambda code: __import__("src.tools.code_engine", fromlist=["SafeCodeSandbox"]).SafeCodeSandbox.execute_snippet(code),
    ))

    registry.register(ToolDefinition(
        name="study_curriculum_book",
        description="Studies an educational textbook (Grades 1-12), extracts concepts, flashcards, and notes.",
        parameters={"book_path": "path to book", "grade_hint": "optional grade level e.g. grade_5"},
        execute_fn=lambda book_path, grade_hint=None: __import__("src.data.curriculum", fromlist=["CurriculumStudyEngine"]).CurriculumStudyEngine(project_root / "data_curriculum").study_book(Path(book_path), grade_hint=grade_hint),
    ))

    registry.register(ToolDefinition(
        name="web_search_duckduckgo",
        description="Searches live web and knowledge graphs using DuckDuckGo with verified citations and links.",
        parameters={"query": "search query string", "max_results": "optional maximum results (default 5)"},
        execute_fn=lambda query, max_results=5: __import__("src.tools.web_search", fromlist=["DuckDuckGoSearchEngine"]).DuckDuckGoSearchEngine.format_search_results(
            query,
            __import__("src.tools.web_search", fromlist=["DuckDuckGoSearchEngine"]).DuckDuckGoSearchEngine.search(query, int(max_results))
        ),
    ))

    registry.register(ToolDefinition(
        name="arxiv_search",
        description="Searches ArXiv for scientific research papers (AI, CS, math, physics, quant-bio).",
        parameters={"query": "scientific paper topic or title", "max_results": "optional maximum results (default 4)"},
        execute_fn=lambda query, max_results=4: __import__("src.tools.free_tools", fromlist=["ArxivSearchEngine"]).ArxivSearchEngine.format_results(
            query,
            __import__("src.tools.free_tools", fromlist=["ArxivSearchEngine"]).ArxivSearchEngine.search(query, int(max_results))
        ),
    ))

    registry.register(ToolDefinition(
        name="wikipedia_summary",
        description="Retrieves full encyclopedic summary and knowledge graph from Wikipedia.",
        parameters={"topic": "concept, entity, or historical topic"},
        execute_fn=lambda topic: __import__("src.tools.free_tools", fromlist=["WikipediaEngine"]).WikipediaEngine.format_summary(
            topic,
            __import__("src.tools.free_tools", fromlist=["WikipediaEngine"]).WikipediaEngine.summary(topic)
        ),
    ))

    registry.register(ToolDefinition(
        name="weather_forecast",
        description="Retrieves live global weather, temperature, wind, and conditions for any city via Open-Meteo.",
        parameters={"city": "city name (e.g. London, Mumbai, Tokyo, New York)"},
        execute_fn=lambda city: __import__("src.tools.free_tools", fromlist=["WeatherEngine"]).WeatherEngine.format_weather(
            city,
            __import__("src.tools.free_tools", fromlist=["WeatherEngine"]).WeatherEngine.get_weather(city)
        ),
    ))

    # --- ADVANCED RESEARCH & CYBERSECURITY TOOLS ---
    registry.register(ToolDefinition(
        name="static_code_audit",
        description="Audits Python source code string for vulnerabilities (command injection, eval, SQLi, deserialization, secrets).",
        parameters={"code": "python code string to audit"},
        execute_fn=lambda code: [f.to_dict() for f in __import__("src.security.code_audit", fromlist=["StaticCodeAuditor"]).StaticCodeAuditor.audit_python_code(code)],
    ))

    registry.register(ToolDefinition(
        name="audit_code_file",
        description="Performs AST security audit on a local code file in the project.",
        parameters={"relative_path": "path relative to project root"},
        execute_fn=lambda relative_path: __import__("src.security.code_audit", fromlist=["StaticCodeAuditor"]).StaticCodeAuditor.audit_file(project_root / relative_path),
    ))

    registry.register(ToolDefinition(
        name="malware_static_inspect",
        description="Performs safe non-executing static analysis of a file (MD5, SHA256, Shannon entropy, strings, headers).",
        parameters={"relative_path": "path relative to project root"},
        execute_fn=lambda relative_path: __import__("src.security.malware_lab", fromlist=["MalwareStaticAnalyzer"]).MalwareStaticAnalyzer.analyze_file(project_root / relative_path).to_dict(),
    ))

    registry.register(ToolDefinition(
        name="ioc_extract",
        description="Extracts indicators of compromise (IPv4, domains, URLs, hashes) from text or reports.",
        parameters={"text": "text containing potential IOCs"},
        execute_fn=lambda text: __import__("src.security.malware_lab", fromlist=["MalwareStaticAnalyzer"]).MalwareStaticAnalyzer.extract_iocs(text),
    ))

    registry.register(ToolDefinition(
        name="yara_rule_generate",
        description="Generates defensive YARA detection rule template from a sample name, sha256 hash, and key strings.",
        parameters={"rule_name": "name of rule", "sha256": "file sha256 hash", "sample_strings": "comma-separated strings"},
        execute_fn=lambda rule_name, sha256, sample_strings: __import__("src.security.malware_lab", fromlist=["MalwareStaticAnalyzer"]).MalwareStaticAnalyzer.generate_yara_rule(
            rule_name=rule_name,
            hashes={"sha256": sha256},
            strings=[s.strip() for s in sample_strings.split(",") if s.strip()]
        ),
    ))

    registry.register(ToolDefinition(
        name="disassemble_bytecode",
        description="Disassembles Python code snippet into bytecode instructions.",
        parameters={"code": "python code snippet"},
        execute_fn=lambda code: __import__("src.security.reverse_lab", fromlist=["ReverseEngineeringToolkit"]).ReverseEngineeringToolkit.disassemble_python_code(code),
    ))

    registry.register(ToolDefinition(
        name="explain_calling_convention",
        description="Provides ABI register mapping and stack frame breakdown for calling conventions (x64_windows_fastcall, system_v_x64, cdecl_x86).",
        parameters={"convention": "convention name (e.g. x64_windows_fastcall)"},
        execute_fn=lambda convention: __import__("src.security.reverse_lab", fromlist=["ReverseEngineeringToolkit"]).ReverseEngineeringToolkit.explain_calling_convention(convention),
    ))

    registry.register(ToolDefinition(
        name="security_lab_sqli",
        description="Demonstrates SQL injection reproduction and parameterized query mitigation in isolated local lab.",
        parameters={"payload": "simulated user input payload"},
        execute_fn=lambda payload: __import__("src.security.lab", fromlist=["IsolatedSecurityLab"]).IsolatedSecurityLab().test_sql_injection_mitigation(payload).to_dict(),
    ))

    registry.register(ToolDefinition(
        name="security_lab_xss",
        description="Demonstrates XSS reflection and HTML entity escaping + CSP mitigation in isolated local lab.",
        parameters={"payload": "simulated HTML/script payload"},
        execute_fn=lambda payload: __import__("src.security.lab", fromlist=["IsolatedSecurityLab"]).IsolatedSecurityLab().test_xss_sanitization_mitigation(payload).to_dict(),
    ))

    registry.register(ToolDefinition(
        name="project_structure_analyze",
        description="Analyzes project directory tree, lines of code, classes, and function symbols.",
        parameters={"max_depth": "optional depth (default 3)"},
        execute_fn=lambda max_depth=3: __import__("src.dev.code_analyzer", fromlist=["ProjectCodeAnalyzer"]).ProjectCodeAnalyzer.analyze_project_structure(project_root, int(max_depth)),
    ))

    registry.register(ToolDefinition(
        name="dependency_inspect",
        description="Inspects project dependencies from requirements.txt.",
        parameters={"filename": "optional requirements file name (default requirements.txt)"},
        execute_fn=lambda filename="requirements.txt": __import__("src.dev.code_analyzer", fromlist=["ProjectCodeAnalyzer"]).ProjectCodeAnalyzer.inspect_dependencies(project_root / filename),
    ))

    # --- BROWSER & AUTONOMOUS AGENT TOOLS ---
    registry.register(ToolDefinition(
        name="browse_web_page",
        description="Headless web browser: fetches an HTTP/HTTPS URL, strips HTML tags/scripts, and extracts clean text, title, and links.",
        parameters={"url": "target webpage URL to browse", "max_chars": "optional character limit (default 4000)"},
        execute_fn=lambda url, max_chars=4000: __import__("src.tools.browser", fromlist=["WebBrowserEngine"]).WebBrowserEngine.browse_url(url, int(max_chars)),
    ))

    registry.register(ToolDefinition(
        name="search_and_browse",
        description="Searches DuckDuckGo and immediately browses and extracts full text from the top search result.",
        parameters={"query": "search topic or query", "max_chars": "optional character limit (default 4000)"},
        execute_fn=lambda query, max_chars=4000: __import__("src.tools.browser", fromlist=["WebBrowserEngine"]).WebBrowserEngine.search_and_browse(query, int(max_chars)),
    ))

    registry.register(ToolDefinition(
        name="run_autonomous_agent",
        description="Launches an autonomous multi-step agent mission with planning, tool chaining, intermediate error recovery, and evidence synthesis.",
        parameters={"goal": "high-level mission goal to accomplish"},
        execute_fn=lambda goal: __import__("src.agents.autonomous_agent", fromlist=["AutonomousWorkAgent"]).AutonomousWorkAgent(registry, verbose=False).execute_mission(goal).to_dict(),
    ))

    # --- GITHUB & GIT AUTOMATION TOOLS ---
    registry.register(ToolDefinition(
        name="github_search_repos",
        description="Searches GitHub repositories by keyword, topic, or language using gh CLI and GitHub API.",
        parameters={"query": "search query or topic", "limit": "optional maximum results (default 5)"},
        execute_fn=lambda query, limit=5: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.search_repositories(query, int(limit)),
    ))

    registry.register(ToolDefinition(
        name="github_get_repo",
        description="Retrieves metadata, stars, forks, and default branch for a GitHub repository (e.g. 'owner/repo').",
        parameters={"repo": "repository full name (owner/repo)"},
        execute_fn=lambda repo: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.get_repo_info(repo),
    ))

    registry.register(ToolDefinition(
        name="github_view_file",
        description="Fetches raw source code or file contents from any public GitHub repository.",
        parameters={"repo": "owner/repo", "file_path": "path to file in repo", "branch": "optional branch (default main)"},
        execute_fn=lambda repo, file_path, branch=None: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.get_file_content(repo, file_path, branch),
    ))

    registry.register(ToolDefinition(
        name="github_list_issues",
        description="Lists open or closed issues from a GitHub repository.",
        parameters={"repo": "owner/repo", "state": "open or closed (default open)", "limit": "optional maximum (default 5)"},
        execute_fn=lambda repo, state="open", limit=5: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.list_issues(repo, state, int(limit)),
    ))

    registry.register(ToolDefinition(
        name="github_list_prs",
        description="Lists open or closed pull requests from a GitHub repository.",
        parameters={"repo": "owner/repo", "state": "open or closed (default open)", "limit": "optional maximum (default 5)"},
        execute_fn=lambda repo, state="open", limit=5: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.list_pull_requests(repo, state, int(limit)),
    ))

    registry.register(ToolDefinition(
        name="git_local_status",
        description="Returns local Git status (current branch, modified files, untracked files).",
        parameters={},
        execute_fn=lambda: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.git_status(project_root),
    ))

    registry.register(ToolDefinition(
        name="git_local_diff",
        description="Returns local uncommitted Git changes and diff summary.",
        parameters={"max_lines": "optional maximum lines (default 100)"},
        execute_fn=lambda max_lines=100: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.git_diff(project_root, int(max_lines)),
    ))

    registry.register(ToolDefinition(
        name="git_local_log",
        description="Returns recent Git commit history.",
        parameters={"limit": "optional maximum commits (default 5)"},
        execute_fn=lambda limit=5: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.git_log(project_root, int(limit)),
    ))

    registry.register(ToolDefinition(
        name="git_local_commit",
        description="Stages modified files and creates a Git commit locally.",
        parameters={"message": "commit message"},
        execute_fn=lambda message: __import__("src.tools.github_engine", fromlist=["GitHubEngine"]).GitHubEngine.git_commit(message, project_root),
    ))
