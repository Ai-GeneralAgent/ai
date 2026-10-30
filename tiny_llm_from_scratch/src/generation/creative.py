"""
Creative Writing & Brainstorming Engine
Generates structured stories, essays, poetry, dialogues, and intuitive analogies
for clear concept explanation and mental modeling.
"""
import random
from typing import Dict, List, Optional


class CreativeEngine:
    """Produces structured creative compositions, analogies, and brainstormed perspectives."""

    ANALOGY_TEMPLATES = {
        "transformer": "A Transformer model is like a master research librarian reading an entire library at once with thousands of bookmarks (attention heads), connecting related sentences across books simultaneously rather than reading word by word like a conveyor belt.",
        "neural_network": "A neural network is like a team of thousands of musicians in an orchestra adjusting the volume and pitch of their instruments after every rehearsal until the symphony perfectly matches the master score.",
        "quantum_mechanics": "Quantum superposition is like a spinning coin on a table: while it spins, it is neither purely heads nor tails, but a blur of probabilities. Only when you stop the coin by observing it does it snap into a single definite state.",
        "gravity": "Gravity in Einstein's general relativity is like placing a heavy bowling ball on a stretched trampoline. The ball curves the fabric beneath it, causing smaller marbles rolled nearby to curve and orbit around it, not because of an invisible tether, but because the path of space itself is bent.",
        "dna": "DNA is like an architectural cookbook written in a four-letter alphabet (A, T, C, G) that contains the precise recipes to construct and repair every organ, protein, and cell in a living organism.",
        "blockchain": "A blockchain is like a shared digital notebook that every member of a town holds an identical copy of. Whenever a new transaction is made, everyone simultaneously writes it down and verifies that it matches all other notebooks.",
    }

    @classmethod
    def get_intuitive_analogy(cls, topic: str) -> Optional[str]:
        """Returns a grounded intuitive analogy for a complex concept."""
        lower = topic.lower().strip().replace("-", " ")
        for key, text in cls.ANALOGY_TEMPLATES.items():
            norm_key = key.replace("_", " ")
            if norm_key in lower or lower in norm_key:
                return text
        return None

    @classmethod
    def format_creative_story(cls, title: str, premise: str, body: str) -> str:
        """Formats a narrative story with clear literary structure."""
        return (
            f"## {title}\n\n"
            f"*{premise}*\n\n"
            "---\n\n"
            f"{body}\n\n"
            "*(Narrative composed with thematic arc & descriptive imagery)*"
        )

    @classmethod
    def format_structured_essay(cls, title: str, thesis: str, sections: List[Dict[str, str]], conclusion: str) -> str:
        """Formats a balanced, analytical essay with introduction, arguments, and conclusion."""
        lines = [
            f"# {title}",
            "",
            f"**Thesis:** {thesis}",
            "",
            "---",
        ]
        for s in sections:
            lines.append(f"\n### {s['heading']}")
            lines.append(s['content'])

        lines.append("\n### Conclusion")
        lines.append(conclusion)
        return "\n".join(lines)
