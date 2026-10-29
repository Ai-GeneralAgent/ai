"""
Claude-Style Artifacts Engine
Detects, formats, and isolates modular artifacts (code snippets, markdown documents,
Mermaid diagrams, and interactive components) with structured metadata and export capability.
"""
from dataclasses import dataclass, field
import re
from typing import List, Optional


@dataclass
class ClaudeArtifact:
    """Represents a discrete, structured Claude-style artifact."""
    identifier: str
    artifact_type: str  # "code", "markdown", "table", "diagram"
    title: str
    content: str
    language: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.identifier,
            "type": self.artifact_type,
            "title": self.title,
            "language": self.language,
            "content": self.content,
        }


class ClaudeArtifactEngine:
    """Extracts, formats, and manages Claude-grade modular artifacts."""

    @classmethod
    def extract_artifacts(cls, text: str) -> List[ClaudeArtifact]:
        """Scans response text and extracts structured code, tables, and diagram artifacts."""
        artifacts: List[ClaudeArtifact] = []
        if not text:
            return artifacts

        # 1. Code blocks & Diagrams
        pattern = r"```([a-zA-Z0-9_\-\+]*)\r?\n([\s\S]*?)```"
        matches = list(re.finditer(pattern, text))
        for idx, match in enumerate(matches, 1):
            lang = match.group(1).lower().strip() or "text"
            content = match.group(2).strip()

            if lang == "mermaid":
                art_type = "diagram"
                title = f"Mermaid Architecture Diagram #{idx}"
            elif lang in ["python", "javascript", "typescript", "html", "css", "c", "cpp", "rust", "sql", "bash"]:
                art_type = "code"
                title = f"{lang.capitalize()} Component #{idx}"
            else:
                art_type = "markdown"
                title = f"Structured Document #{idx}"

            artifacts.append(ClaudeArtifact(
                identifier=f"artifact-{idx}",
                artifact_type=art_type,
                title=title,
                content=content,
                language=lang if art_type == "code" else None,
            ))

        # 2. Markdown Tables (if not already inside a code block)
        table_pattern = r"(\|.+?\|\n\|[-:\s|]+?\|\n(?:\|.+?\|\n?)+)"
        table_matches = re.findall(table_pattern, text)
        for idx, t_content in enumerate(table_matches, len(artifacts) + 1):
            # Verify table isn't inside any already matched code snippet
            if not any(t_content in a.content for a in artifacts):
                artifacts.append(ClaudeArtifact(
                    identifier=f"artifact-{idx}",
                    artifact_type="table",
                    title=f"Data Table #{idx}",
                    content=t_content.strip(),
                    language=None,
                ))

        return artifacts

    @classmethod
    def format_artifact_display(cls, artifact: ClaudeArtifact) -> str:
        """Renders a Claude-style visual terminal artifact card with universal terminal compatibility."""
        title_str = f" [Artifact: {artifact.title}] ({artifact.artifact_type.upper()}) "
        width = max(64, len(title_str) + 6)
        h_pad = "-" * (width - len(title_str) - 2)

        header = f"+--{title_str}{h_pad}+"
        divider = "+" + ("-" * (width + 1)) + "+"
        footer = "+" + ("-" * (width + 1)) + "+"

        lines = [header]
        if artifact.language:
            lines.append(f"| Language: {artifact.language}")
            lines.append(divider)
        for line in artifact.content.splitlines():
            lines.append(f"| {line}")
        lines.append(footer)
        return "\n".join(lines)
