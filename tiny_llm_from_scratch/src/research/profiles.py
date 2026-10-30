"""
Advanced Research Mode - Execution Profiles and Scope Management
Provides 8 distinct profiles for deep technical engineering, research, and analysis
without overly broad keyword-based refusals.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ResearchProfile(str, Enum):
    NORMAL_ASSISTANT = "Normal Assistant"
    ADVANCED_DEVELOPMENT = "Advanced Development"
    CYBERSECURITY_RESEARCH = "Cybersecurity Research"
    AUTHORIZED_RED_TEAM_LAB = "Authorized Red Team Lab"
    REVERSE_ENGINEERING_LAB = "Reverse Engineering Lab"
    MALWARE_ANALYSIS_LAB = "Malware Analysis Lab"
    CTF_SECURITY_CHALLENGES = "CTF and Security Challenges"
    AI_RESEARCH_MODE = "AI Research Mode"


@dataclass
class ProfileDefinition:
    profile: ResearchProfile
    title: str
    description: str
    scope: str
    capabilities: List[str]
    allowed_tool_categories: Set[str]
    allow_external_network: bool = False
    requires_explicit_lab_target: bool = False
    safety_boundary: str = ""


PROFILES: Dict[ResearchProfile, ProfileDefinition] = {
    ResearchProfile.NORMAL_ASSISTANT: ProfileDefinition(
        profile=ResearchProfile.NORMAL_ASSISTANT,
        title="Normal Assistant",
        description="Everyday helpful, friendly, and concise conversational assistant.",
        scope="General conversational queries, basic factual Q&A, and casual assistance.",
        capabilities=[
            "Natural casual dialogue",
            "Factual question answering",
            "Concise summaries",
            "General math and calculator access",
        ],
        allowed_tool_categories={"math", "datetime", "rag_basic"},
        allow_external_network=False,
        safety_boundary="Standard friendly persona, concise answers.",
    ),
    ResearchProfile.ADVANCED_DEVELOPMENT: ProfileDefinition(
        profile=ResearchProfile.ADVANCED_DEVELOPMENT,
        title="Advanced Development Mode",
        description="Full-stack software engineering, architecture, code review, debugging, and testing.",
        scope="Multi-language code understanding (Python, C/C++, Rust, Go, JS/TS, C#, Java), architecture, AST inspection, unit test generation, dependency review, and performance optimization.",
        capabilities=[
            "Multi-file project structure analysis",
            "AST-based syntax and logic inspection",
            "Root-cause bug localization",
            "Unit and integration test synthesis",
            "Dependency and package security inspection",
            "Refactoring and design pattern evaluation",
            "Git, CI/CD, and build system configuration",
        ],
        allowed_tool_categories={"math", "datetime", "dev_tools", "code_engine", "rag"},
        allow_external_network=False,
        safety_boundary="Read-only filesystem inspection by default; user approval required for file mutations.",
    ),
    ResearchProfile.CYBERSECURITY_RESEARCH: ProfileDefinition(
        profile=ResearchProfile.CYBERSECURITY_RESEARCH,
        title="Cybersecurity Research Mode",
        description="Defensive security analysis, vulnerability research, threat modeling, and secure architecture.",
        scope="In-depth technical analysis of web vulnerabilities (OWASP Top 10), network protocols, system internals (Windows/Linux), cryptography verification, and defensive mitigation without keyword refusals.",
        capabilities=[
            "Static source code security audit (AST taint, insecure APIs, injection flaws)",
            "Vulnerability root cause and mechanics explanation (SQLi, XSS, SSRF, CSRF, IDOR)",
            "Cryptographic algorithm and key management review",
            "Threat modeling and attack surface mapping (STRIDE)",
            "Defensive remediation and secure patch design",
            "Security header and policy evaluation (CSP, HSTS, CORS)",
        ],
        allowed_tool_categories={"math", "datetime", "dev_tools", "security_audit", "rag"},
        allow_external_network=False,
        safety_boundary="Defensive analysis, secure code reviews, and theoretical explanations; no unauthorized targeting.",
    ),
    ResearchProfile.AUTHORIZED_RED_TEAM_LAB: ProfileDefinition(
        profile=ResearchProfile.AUTHORIZED_RED_TEAM_LAB,
        title="Authorized Red Team Lab",
        description="Controlled threat emulation and security boundary testing in explicitly authorized local labs.",
        scope="Methodology for authorized penetration testing, vulnerability reproduction on local mock targets, proof-of-concept verification inside isolated sandboxes.",
        capabilities=[
            "Authorized lab vulnerability reproduction",
            "Mock test application security verification",
            "Security control bypass mechanism analysis (why controls fail)",
            "Boundary and access control testing",
            "Professional penetration testing reporting and remediation guidance",
        ],
        allowed_tool_categories={"math", "datetime", "security_lab", "security_audit", "dev_tools"},
        allow_external_network=False,
        requires_explicit_lab_target=True,
        safety_boundary="Strict localhost/127.0.0.1 allowlist only. No arbitrary internet scanning, credential theft, or stealth persistence.",
    ),
    ResearchProfile.REVERSE_ENGINEERING_LAB: ProfileDefinition(
        profile=ResearchProfile.REVERSE_ENGINEERING_LAB,
        title="Reverse Engineering Lab",
        description="Binary format inspection, disassembly, decompilation concepts, and software compatibility research.",
        scope="Executable format analysis (PE/COFF, ELF), bytecodes, assembly fundamentals (x86, x64, ARM), calling conventions, and control flow graph theory.",
        capabilities=[
            "PE header and section analysis (.text, .rdata, .data, .rsrc)",
            "ELF header and symbol table inspection",
            "Python bytecode disassembly (dis module)",
            "x86/x64/ARM instruction decode and calling convention tracer",
            "Control flow reconstruction and basic block analysis",
            "Safe educational samples and CTF binary puzzle analysis",
        ],
        allowed_tool_categories={"math", "datetime", "reverse_tools", "dev_tools"},
        allow_external_network=False,
        safety_boundary="Static and isolated inspection of authorized files only; no weaponization.",
    ),
    ResearchProfile.MALWARE_ANALYSIS_LAB: ProfileDefinition(
        profile=ResearchProfile.MALWARE_ANALYSIS_LAB,
        title="Malware Analysis Lab",
        description="Safe static artifact inspection, detection engineering, and incident response analysis.",
        scope="Static analysis of benign or authorized samples, cryptographic hashing, entropy calculation, string extraction, YARA rule generation, and IOC extraction.",
        capabilities=[
            "Cryptographic file hashing (MD5, SHA1, SHA256)",
            "Shannon entropy analysis (detecting packed or encrypted sections)",
            "ASCII and Unicode string extraction",
            "Automated YARA rule synthesis",
            "Indicator of Compromise (IOC) extraction (IPs, hashes, domains)",
            "Defensive detection engineering and behavioral classification",
        ],
        allowed_tool_categories={"math", "datetime", "malware_lab", "security_audit"},
        allow_external_network=False,
        safety_boundary="Strictly static analysis on local samples. No execution of destructive payloads.",
    ),
    ResearchProfile.CTF_SECURITY_CHALLENGES: ProfileDefinition(
        profile=ResearchProfile.CTF_SECURITY_CHALLENGES,
        title="CTF & Security Challenges Mode",
        description="Educational breakdowns, puzzle solving, and walk-throughs for Capture The Flag competitions.",
        scope="Cryptography puzzles, web security challenges, steganography, forensics, and binary exploitation theory in competitive learning contexts.",
        capabilities=[
            "Classic cipher and modern crypto puzzle cryptanalysis (RSA, XOR, Diffie-Hellman)",
            "Web challenge logic flaw identification",
            "Forensic artifact carving and metadata extraction",
            "Binary exploitation concepts (stack frames, buffer boundaries, format strings)",
            "Step-by-step educational writeups and hint ladders",
        ],
        allowed_tool_categories={"math", "datetime", "reverse_tools", "security_lab", "dev_tools"},
        allow_external_network=False,
        safety_boundary="Educational and competition challenge contexts only.",
    ),
    ResearchProfile.AI_RESEARCH_MODE: ProfileDefinition(
        profile=ResearchProfile.AI_RESEARCH_MODE,
        title="AI Research Mode",
        description="Transformer neural architectures, mathematical foundations, training dynamics, and optimization.",
        scope="Deep analysis of modern LLM internals: RMSNorm, SwiGLU, RoPE/YaRN, GQA, KV-cache management, Muon/AdamW optimizers, tokenization economics, distillation, and RLVR.",
        capabilities=[
            "Transformer layer parameter and FLOP calculation",
            "RoPE frequency and long-context YaRN scaling calculations",
            "StreamingLLM attention sink and KV-cache memory profiling",
            "Muon vs AdamW optimizer dynamics analysis",
            "Byte-level BPE tokenizer vocabulary and compression analysis",
            "Distillation and self-verification curriculum design",
        ],
        allowed_tool_categories={"math", "datetime", "ai_tools", "dev_tools", "rag"},
        allow_external_network=False,
        safety_boundary="Theoretical and empirical machine learning research.",
    ),
}


class ResearchModeManager:
    """Manages active research profiles, validates capabilities, and prevents silent authorization leaks."""

    def __init__(self, initial_profile: ResearchProfile = ResearchProfile.ADVANCED_DEVELOPMENT):
        self.active_profile: ResearchProfile = initial_profile
        self.authorized_target_scope: Set[str] = {"127.0.0.1", "localhost", "::1", "test.local"}
        self.execution_history: List[Dict[str, Any]] = []

    def set_profile(self, profile: ResearchProfile) -> ProfileDefinition:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}. Available: {[p.value for p in PROFILES]}")
        self.active_profile = profile
        return PROFILES[profile]

    def set_profile_by_name(self, name_or_number: str) -> ProfileDefinition:
        """Sets profile by index (1-8) or case-insensitive name match."""
        s = name_or_number.strip().lower()
        # Number mapping
        profiles_list = list(ResearchProfile)
        try:
            idx = int(s)
            if 1 <= idx <= len(profiles_list):
                return self.set_profile(profiles_list[idx - 1])
        except ValueError:
            pass

        for p in profiles_list:
            if s in p.value.lower() or s == p.name.lower():
                return self.set_profile(p)
        raise ValueError(f"Could not match profile '{name_or_number}'. Options: {[p.value for p in profiles_list]}")

    def get_active_definition(self) -> ProfileDefinition:
        return PROFILES[self.active_profile]

    def validate_action(self, category: str, target: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validates whether an action is permitted within the current profile's configured scope.
        Does not allow silent authorization outside configured scope.
        """
        defn = self.get_active_definition()
        if category not in defn.allowed_tool_categories and "all" not in defn.allowed_tool_categories:
            return False, f"Action category '{category}' is outside scope of active profile '{defn.title}'."

        if target is not None:
            clean_target = target.lower().split(":")[0].strip()
            if clean_target not in self.authorized_target_scope:
                return False, f"Target '{target}' is not in authorized lab scope {self.authorized_target_scope}. Arbitrary external network targets are forbidden."

        return True, None

    def format_banner(self) -> str:
        d = self.get_active_definition()
        lines = [
            f"=== [PROFILE: {d.title.upper()}] ===",
            f"Scope:        {d.scope}",
            f"Boundary:     {d.safety_boundary}",
            f"Capabilities: {', '.join(d.capabilities[:4])}...",
            f"Tool Access:  {', '.join(d.allowed_tool_categories)}",
            f"Target Scope: {', '.join(self.authorized_target_scope)}",
            "=" * 50,
        ]
        return "\n".join(lines)

