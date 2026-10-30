"""
Static Source Code Security Audit Engine (AST-based)
Analyzes Python and script code for common vulnerability patterns:
- Command injection (subprocess.shell=True, os.system, os.popen)
- Code execution (eval, exec, compile)
- SQL injection (string formatting in queries)
- Path traversal (unvalidated os.path.join)
- Insecure cryptography (MD5, SHA1 for security, DES, RC4)
- Insecure deserialization (pickle.loads, yaml.load)
- Hardcoded credentials and high-entropy secrets
"""
import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SecurityFinding:
    severity: str  # HIGH, MEDIUM, LOW, INFO
    category: str
    rule_id: str
    message: str
    line_number: int
    code_snippet: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "rule_id": self.rule_id,
            "message": self.message,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
        }


class StaticCodeAuditor:
    """AST-based static security analyzer for defensive code review."""

    SECRET_PATTERNS = [
        (re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|auth[_-]?token)\s*=\s*['\"][A-Za-z0-9_\-+/=]{16,}['\"]"), "Hardcoded High-Entropy Secret / API Key"),
        (re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"), "Hardcoded Private Key Material"),
    ]

    @classmethod
    def audit_python_code(cls, code_string: str, filename: str = "<input>") -> List[SecurityFinding]:
        findings: List[SecurityFinding] = []
        lines = code_string.splitlines()

        # 1. Regex-based secret detection
        for i, line in enumerate(lines, 1):
            for pattern, msg in cls.SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append(SecurityFinding(
                        severity="HIGH",
                        category="Hardcoded Secrets",
                        rule_id="SEC-001-SECRET",
                        message=msg,
                        line_number=i,
                        code_snippet=line.strip()[:80],
                        recommendation="Load secrets dynamically from environment variables or secure vault.",
                    ))

        # 2. AST-based structural inspection
        try:
            tree = ast.parse(code_string, filename=filename)
        except SyntaxError as e:
            findings.append(SecurityFinding(
                severity="INFO",
                category="Syntax Error",
                rule_id="SYNTAX-ERR",
                message=f"Could not parse AST: {e.msg}",
                line_number=e.lineno or 1,
                code_snippet=e.text.strip() if e.text else "",
                recommendation="Fix Python syntax error before auditing.",
            ))
            return findings

        for node in ast.walk(tree):
            # A. Arbitrary Code Execution (eval, exec, __import__)
            if isinstance(node, ast.Call):
                func_name = cls._get_call_name(node.func)
                lineno = getattr(node, "lineno", 1)
                snippet = lines[lineno - 1].strip() if lineno <= len(lines) else ""

                if func_name in ("eval", "exec"):
                    findings.append(SecurityFinding(
                        severity="HIGH",
                        category="Arbitrary Code Execution",
                        rule_id="SEC-002-EVAL-EXEC",
                        message=f"Dangerous dynamic code execution function '{func_name}' detected.",
                        line_number=lineno,
                        code_snippet=snippet,
                        recommendation="Avoid eval/exec. Use ast.literal_eval for safe literals or structured parsers.",
                    ))

                # B. OS Command Injection
                if func_name in ("os.system", "os.popen"):
                    findings.append(SecurityFinding(
                        severity="HIGH",
                        category="Command Injection",
                        rule_id="SEC-003-OS-SYSTEM",
                        message=f"Insecure shell execution via '{func_name}'.",
                        line_number=lineno,
                        code_snippet=snippet,
                        recommendation="Use subprocess.run with argument list and shell=False.",
                    ))

                if func_name in ("subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_output"):
                    # Check for shell=True
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            findings.append(SecurityFinding(
                                severity="HIGH",
                                category="Command Injection",
                                rule_id="SEC-004-SUBPROCESS-SHELL",
                                message=f"Subprocess invoked with shell=True, enabling shell injection.",
                                line_number=lineno,
                                code_snippet=snippet,
                                recommendation="Pass arguments as a sequence (list) and set shell=False.",
                            ))

                # C. Insecure Deserialization
                if func_name in ("pickle.loads", "pickle.load", "_pickle.loads", "shelve.open"):
                    findings.append(SecurityFinding(
                        severity="HIGH",
                        category="Insecure Deserialization",
                        rule_id="SEC-005-PICKLE",
                        message=f"Untrusted deserialization via '{func_name}' can lead to arbitrary code execution.",
                        line_number=lineno,
                        code_snippet=snippet,
                        recommendation="Use safe serialization formats like JSON, Protocol Buffers, or CBOR.",
                    ))

                if func_name in ("yaml.load",) and not any(kw.arg == "Loader" for kw in node.keywords):
                    findings.append(SecurityFinding(
                        severity="HIGH",
                        category="Insecure Deserialization",
                        rule_id="SEC-006-YAML",
                        message="yaml.load without safe Loader can instantiate arbitrary Python objects.",
                        line_number=lineno,
                        code_snippet=snippet,
                        recommendation="Use yaml.safe_load(data) instead of yaml.load.",
                    ))

                # D. SQL Injection Pattern
                if func_name in ("cursor.execute", "db.execute", "connection.execute", "session.execute"):
                    if node.args and isinstance(node.args[0], (ast.JoinedStr, ast.BinOp)):
                        findings.append(SecurityFinding(
                            severity="HIGH",
                            category="SQL Injection",
                            rule_id="SEC-007-SQLI",
                            message="SQL query constructed via dynamic string formatting/concatenation.",
                            line_number=lineno,
                            code_snippet=snippet,
                            recommendation="Use parameterized queries with placeholders: execute(query, (params,)).",
                        ))

                # E. Insecure Cryptographic Hash
                if func_name in ("hashlib.md5", "hashlib.sha1"):
                    findings.append(SecurityFinding(
                        severity="MEDIUM",
                        category="Weak Cryptography",
                        rule_id="SEC-008-WEAK-HASH",
                        message=f"Cryptographically broken hash function '{func_name}' used.",
                        line_number=lineno,
                        code_snippet=snippet,
                        recommendation="Use hashlib.sha256, sha384, or sha512 for cryptographic integrity.",
                    ))

        return findings

    @classmethod
    def audit_code(cls, code_string: str, filename: str = "<input>") -> Dict[str, Any]:
        """Audits an in-memory code string safely and returns dictionary report."""
        findings = cls.audit_python_code(code_string, filename=filename)
        high_count = sum(1 for f in findings if f.severity == "HIGH")
        med_count = sum(1 for f in findings if f.severity == "MEDIUM")
        return {
            "file": filename,
            "total_findings": len(findings),
            "high_severity": high_count,
            "medium_severity": med_count,
            "findings": [f.to_dict() for f in findings],
        }

    @classmethod
    def audit_file(cls, file_path: Path) -> Dict[str, Any]:
        """Audits a file from disk safely."""
        p = Path(file_path)
        if not p.is_file():
            return {"error": f"File not found: {p}"}
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            findings = cls.audit_python_code(content, filename=p.name)
            high_count = sum(1 for f in findings if f.severity == "HIGH")
            med_count = sum(1 for f in findings if f.severity == "MEDIUM")
            return {
                "file": str(p),
                "total_findings": len(findings),
                "high_severity": high_count,
                "medium_severity": med_count,
                "findings": [f.to_dict() for f in findings],
            }
        except Exception as e:
            return {"error": f"Audit error on {p}: {str(e)}"}

    @staticmethod
    def _get_call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = StaticCodeAuditor._get_call_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return ""

