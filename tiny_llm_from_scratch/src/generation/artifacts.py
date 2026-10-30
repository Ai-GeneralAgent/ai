"""
Structured Artifacts Engine
Detects, formats, and isolates modular artifacts (code snippets, markdown documents,
Mermaid diagrams, and interactive components) with structured metadata and export capability.
"""
from dataclasses import dataclass, field
import re
from typing import List, Optional


@dataclass
class Artifact:
    """Represents a discrete, structured artifact."""
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


# Backward compatibility alias
ClaudeArtifact = Artifact


class ArtifactEngine:
    """Extracts, formats, and manages modular artifacts."""

    @classmethod
    def extract_artifacts(cls, text: str) -> List[Artifact]:
        """Scans response text and extracts structured code, tables, and diagram artifacts."""
        artifacts: List[Artifact] = []
        if not text:
            return artifacts

        # 1. Code blocks & Diagrams
        pattern = r"```([a-zA-Z0-9_\-\+]*)\r?\n([\s\S]*?)```"
        matches = list(re.finditer(pattern, text))
        for idx, match in enumerate(matches, 1):
            lang = match.group(1).lower().strip() or "text"
            content = match.group(2).strip()

            if lang in ["mermaid", "graph"]:
                art_type = "diagram"
                title = f"Diagram #{idx}"
            elif lang in ["python", "py", "javascript", "js", "html", "css", "sql", "bash", "sh", "c", "cpp"]:
                art_type = "code"
                title = f"{lang.capitalize()} Component #{idx}"
            elif lang in ["markdown", "md"]:
                art_type = "markdown"
                title = f"Document #{idx}"
            else:
                art_type = "code"
                title = f"Code Snippet #{idx}"

            artifacts.append(Artifact(
                identifier=f"artifact_{idx:03d}",
                artifact_type=art_type,
                title=title,
                content=content,
                language=lang,
            ))

        # 2. Markdown tables
        table_pattern = r"(?:\|[^\n]+\|\r?\n){2,}"
        for t_idx, t_match in enumerate(re.finditer(table_pattern, text), len(artifacts) + 1):
            table_content = t_match.group(0).strip()
            if table_content:
                artifacts.append(Artifact(
                    identifier=f"artifact_{t_idx:03d}",
                    artifact_type="table",
                    title=f"Data Table #{t_idx}",
                    content=table_content,
                    language="markdown",
                ))

        return artifacts

    @classmethod
    def format_artifact_display(cls, artifact: Artifact) -> str:
        """Renders a visual terminal artifact card with universal ASCII character compatibility."""
        border_width = 65
        title_tag = f" [Artifact: {artifact.title}] ({artifact.artifact_type.upper()}) "
        top_bar = f"+--{title_tag}"
        if len(top_bar) < border_width:
            top_bar += "-" * (border_width - len(top_bar)) + "+"
        else:
            top_bar += "-+"

        bottom_bar = "+" + "-" * (border_width - 1) + "+"

        lines = [top_bar]
        if artifact.language:
            lines.append(f"| Language: {artifact.language}")
            lines.append("+" + "-" * (border_width - 1) + "+")

        for line in artifact.content.splitlines():
            lines.append(f"| {line}")

        lines.append(bottom_bar)
        return "\n".join(lines)


# Backward compatibility alias
ClaudeArtifactEngine = ArtifactEngine
