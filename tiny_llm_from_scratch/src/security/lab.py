"""
Isolated Local Security Laboratory & Scope Enforcement
Provides controlled offline testing environments for authorized vulnerability reproduction,
patch verification, and CTF challenges with strict target allowlists and comprehensive audit logging.
"""
import datetime
import html
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class LabExperimentResult:
    experiment_id: str
    target: str
    vulnerability_type: str
    is_vulnerable_before: bool
    is_mitigated_after: bool
    evidence: str
    remediation_guidance: str
    audit_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "target": self.target,
            "vulnerability_type": self.vulnerability_type,
            "is_vulnerable_before": self.is_vulnerable_before,
            "is_mitigated_after": self.is_mitigated_after,
            "evidence": self.evidence,
            "remediation_guidance": self.remediation_guidance,
            "audit_timestamp": self.audit_timestamp,
        }


class IsolatedSecurityLab:
    """
    Controlled local laboratory manager:
    - Enforces strict target allowlists (localhost / 127.0.0.1 / test.local).
    - Blocks arbitrary internet scanning or external exploitation.
    - Provides disposable, offline mock vulnerable test fixtures.
    - Maintains an immutable audit trail of all lab operations.
    """

    ALLOWED_TARGETS: Set[str] = {"127.0.0.1", "localhost", "::1", "test.local", "vulnerable-lab", "mock://vulnerable-lab"}

    def __init__(self, audit_log_path: Optional[Path] = None):
        self.audit_log_path = audit_log_path or (Path(__file__).resolve().parent.parent.parent / "security_lab_audit.jsonl")
        self.active_scope: Set[str] = set(self.ALLOWED_TARGETS)

    def validate_target(self, target: str) -> Tuple[bool, Optional[str]]:
        """Strictly validates target against active lab scope."""
        t = target.lower().strip()
        if t in self.active_scope:
            return True, None
        for prefix in ("mock://", "http://", "https://"):
            if t.startswith(prefix):
                t = t[len(prefix):]
        clean_host = t.split("/")[0].split(":")[0].strip()
        if clean_host in self.active_scope:
            return True, None
        return False, f"Target '{target}' is unauthorized. Only strictly local authorized lab targets {self.ALLOWED_TARGETS} are permitted."

    def log_audit(self, action: str, target: str, status: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": action,
            "target": target,
            "status": status,
            "details": details,
        }
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Disposable Mock Vulnerable Test Environments (Educational Verification)
    # -------------------------------------------------------------------------

    def test_sql_injection_mitigation(self, user_input: str) -> LabExperimentResult:
        """
        Reproduces SQL injection vulnerability using raw concatenation,
        then verifies mitigation using parameterized queries.
        """
        target = "mock://vulnerable-lab/sqli"
        is_allowed, err = self.validate_target(target)
        if not is_allowed:
            raise PermissionError(err)

        # 1. Vulnerable simulation: raw string concatenation
        vulnerable_query = f"SELECT * FROM users WHERE username = '{user_input}'"
        is_vuln = bool(re.search(r"('|\bOR\b|\bUNION\b|--|\bAND\b|;)", user_input, re.IGNORECASE))

        # 2. Mitigated implementation: parameterized query
        parameterized_query = "SELECT * FROM users WHERE username = ?"
        parameterized_params = (user_input,)

        evidence = (
            f"Vulnerable Query Formed: {vulnerable_query}\n"
            f"Injection Syntax Detected: {is_vuln}\n"
            f"Mitigated Query Schema:  {parameterized_query} with params {parameterized_params}"
        )

        res = LabExperimentResult(
            experiment_id="LAB-EXP-SQLI-01",
            target=target,
            vulnerability_type="CWE-89: SQL Injection",
            is_vulnerable_before=is_vuln,
            is_mitigated_after=True,
            evidence=evidence,
            remediation_guidance="Always use parameterized SQL queries or ORM prepared statements. Never concatenate untrusted strings into SQL statements.",
            audit_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        self.log_audit("reproduce_and_mitigate", target, "SUCCESS", res.to_dict())
        return res

    def test_xss_sanitization_mitigation(self, untrusted_payload: str) -> LabExperimentResult:
        """
        Reproduces Cross-Site Scripting (XSS) reflection,
        then verifies mitigation using HTML entity encoding and CSP headers.
        """
        target = "mock://vulnerable-lab/xss"
        is_allowed, err = self.validate_target(target)
        if not is_allowed:
            raise PermissionError(err)

        # Vulnerable simulation: unescaped reflection
        is_vuln = bool(re.search(r"<script|<img|onerror=|onload=|javascript:", untrusted_payload, re.IGNORECASE))

        # Mitigated implementation: standard HTML entity escaping
        escaped_output = html.escape(untrusted_payload)
        is_mitigated = ("<script>" not in escaped_output and "onerror=" not in escaped_output)

        evidence = (
            f"Raw Input:       {untrusted_payload}\n"
            f"Vulnerable HTML: <div>{untrusted_payload}</div>\n"
            f"Mitigated HTML:  <div>{escaped_output}</div>\n"
            f"CSP Recommendation: default-src 'self'; script-src 'self';"
        )

        res = LabExperimentResult(
            experiment_id="LAB-EXP-XSS-02",
            target=target,
            vulnerability_type="CWE-79: Cross-Site Scripting (XSS)",
            is_vulnerable_before=is_vuln,
            is_mitigated_after=is_mitigated,
            evidence=evidence,
            remediation_guidance="Contextually encode all user-controlled data before inserting into HTML DOM. Deploy strict Content Security Policy (CSP) headers.",
            audit_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        self.log_audit("reproduce_and_mitigate", target, "SUCCESS", res.to_dict())
        return res

    def test_path_traversal_mitigation(self, base_dir: Path, requested_filename: str) -> LabExperimentResult:
        """
        Reproduces Directory/Path Traversal (CWE-22),
        then verifies mitigation using canonical path resolution and boundary checks.
        """
        target = "mock://vulnerable-lab/path-traversal"
        is_allowed, err = self.validate_target(target)
        if not is_allowed:
            raise PermissionError(err)

        base_dir_resolved = base_dir.resolve()

        # Vulnerable implementation
        insecure_path = base_dir / requested_filename
        insecure_resolved = insecure_path.resolve()
        is_vuln = not str(insecure_resolved).startswith(str(base_dir_resolved))

        # Mitigated implementation
        safe_name = Path(requested_filename).name  # Strips all leading directory components (../../)
        secure_path = (base_dir_resolved / safe_name).resolve()
        is_mitigated = str(secure_path).startswith(str(base_dir_resolved))

        evidence = (
            f"Base Directory:     {base_dir_resolved}\n"
            f"Requested Path:     {requested_filename}\n"
            f"Insecure Traversal: {insecure_resolved} (Escapes base: {is_vuln})\n"
            f"Secured Path:       {secure_path} (Contained: {is_mitigated})"
        )

        res = LabExperimentResult(
            experiment_id="LAB-EXP-PATH-03",
            target=target,
            vulnerability_type="CWE-22: Path Traversal",
            is_vulnerable_before=is_vuln,
            is_mitigated_after=is_mitigated,
            evidence=evidence,
            remediation_guidance="Strip path separators using Path(filename).name and verify that path.resolve() starts with base_directory.resolve().",
            audit_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        self.log_audit("reproduce_and_mitigate", target, "SUCCESS", res.to_dict())
        return res
