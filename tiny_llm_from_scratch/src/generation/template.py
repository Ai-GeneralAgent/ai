from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChatTemplate:
    """Versioned conversation formatting template (v2.0)."""

    def __init__(self, template_type: str = "standard"):
        self.template_type = template_type

    def format(self, messages: List[ChatMessage]) -> str:
        formatted_turns = []
        for msg in messages:
            if msg.role == "system":
                formatted_turns.append(f"### System:\n{msg.content.strip()}\n")
            elif msg.role == "user":
                formatted_turns.append(f"### User:\n{msg.content.strip()}\n")
            elif msg.role == "assistant":
                formatted_turns.append(f"### Assistant:\n{msg.content.strip()}\n")
            elif msg.role == "tool":
                formatted_turns.append(f"### Tool [{msg.metadata.get('tool_name', 'call')}]:\n{msg.content.strip()}\n")

        # Append assistant prompt trigger
        formatted_turns.append("### Assistant:\n")
        return "\n".join(formatted_turns)

    def extract_assistant_response(self, raw_output: str) -> str:
        """Strips future turn triggers if the model generates them."""
        for stop in ["### User:", "### System:", "### Tool"]:
            if stop in raw_output:
                raw_output = raw_output.split(stop)[0]
        return raw_output.strip()

