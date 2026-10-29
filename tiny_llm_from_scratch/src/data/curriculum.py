"""
Curriculum Study Engine
Simulates human-like learning progression across Grades 1 to 12.
Extracts core concepts, definitions, formulas, and generates Q&A study notes,
flashcards, and instruction-tuning datasets from educational textbooks and books.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .pipeline import CorpusPipeline
from ..rag.pipeline import LocalRAGPipeline, DocumentChunk


class CurriculumStudyEngine:
    """
    Ingests and studies textbooks systematically like a human student:
    Reading -> Concept Extraction -> Flashcards/Q&A Generation -> RAG Grounding -> Study Notes.
    """

    GRADE_LEVELS = {
        "grade_1": {"name": "Grade 1 (Foundations)", "difficulty": 1},
        "grade_2": {"name": "Grade 2 (Basic Reading & Math)", "difficulty": 2},
        "grade_3": {"name": "Grade 3 (Early Elementary)", "difficulty": 3},
        "grade_4": {"name": "Grade 4 (Elementary Science & Social)", "difficulty": 4},
        "grade_5": {"name": "Grade 5 (Upper Elementary)", "difficulty": 5},
        "grade_6": {"name": "Grade 6 (Middle School Science & Math)", "difficulty": 6},
        "grade_7": {"name": "Grade 7 (Pre-Algebra & Life Science)", "difficulty": 7},
        "grade_8": {"name": "Grade 8 (Algebra & Physical Science)", "difficulty": 8},
        "grade_9": {"name": "Grade 9 (High School Biology & Algebra I)", "difficulty": 9},
        "grade_10": {"name": "Grade 10 (Geometry & Chemistry)", "difficulty": 10},
        "grade_11": {"name": "Grade 11 (Physics & Algebra II / Pre-Calc)", "difficulty": 11},
        "grade_12": {"name": "Grade 12 (Advanced Placement / Calculus / CS)", "difficulty": 12},
    }

    def __init__(self, output_dir: Optional[Path] = None, rag_pipeline: Optional[LocalRAGPipeline] = None):
        self.output_dir = output_dir or Path("data_curriculum")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rag = rag_pipeline
        self.pipeline = CorpusPipeline()

    def detect_grade_level(self, text: str, hint: Optional[str] = None) -> str:
        """Infers appropriate grade level based on vocabulary and structure."""
        if hint and hint.lower() in self.GRADE_LEVELS:
            return hint.lower()

        lower = text.lower()
        # Look for explicit grade markers
        match = re.search(r"\b(class|grade|standard|std)\s*([1-9]|1[0-2])\b", lower)
        if match:
            grade_num = int(match.group(2))
            return f"grade_{grade_num}"

        # Heuristic vocabulary complexity analysis
        advanced_keywords = [
            "derivative", "integral", "quantum", "electromagnetism", "thermodynamics",
            "stoichiometry", "mitochondria", "photosynthesis", "quadratic", "polynomial",
            "homeostasis", "asymptote", "vector", "equilibrium", "algorithm"
        ]
        intermediate_keywords = [
            "fraction", "decimal", "ecosystem", "revolution", "atmosphere",
            "gravity", "evaporation", "constitution", "skeleton", "molecule"
        ]

        adv_count = sum(1 for k in advanced_keywords if k in lower)
        int_count = sum(1 for k in intermediate_keywords if k in lower)

        if adv_count >= 3:
            return "grade_11" if adv_count < 6 else "grade_12"
        elif int_count >= 3:
            return "grade_7" if int_count < 6 else "grade_9"
        else:
            return "grade_4"

    def extract_key_concepts(self, text: str) -> List[Dict[str, str]]:
        """Extracts definitions, laws, and core concepts like a student highlighting a textbook."""
        concepts = []
        seen_terms = set()

        # Pattern 1: Term is defined as / refers to / is called
        patterns = [
            r"([A-Z][a-zA-Z\s]{2,30})\s+is\s+(?:defined\s+as|known\s+as|called)\s+([^.\n]+(?:\.[^.\n]+)?)",
            r"([A-Z][a-zA-Z\s]{2,30})\s+refers\s+to\s+([^.\n]+(?:\.[^.\n]+)?)",
            r"(?:The\s+)?([A-Z][a-zA-Z\s]{2,30})\s+(?:states\s+that|law\s+states)\s+([^.\n]+(?:\.[^.\n]+)?)",
        ]

        for pat in patterns:
            for match in re.finditer(pat, text):
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if len(term.split()) <= 4 and term.lower() not in seen_terms and len(definition) > 15:
                    seen_terms.add(term.lower())
                    concepts.append({
                        "term": term,
                        "definition": definition,
                        "type": "definition",
                    })
                    if len(concepts) >= 50:
                        break

        # Fallback: Extract heading-based summary concepts if regex is sparse
        if len(concepts) < 5:
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]
            for p in paragraphs[:15]:
                first_sent = p.split(".")[0].strip()
                if 20 < len(first_sent) < 120:
                    concepts.append({
                        "term": first_sent[:40] + "...",
                        "definition": p[:250],
                        "type": "summary",
                    })

        return concepts

    def generate_study_flashcards(self, concepts: List[Dict[str, str]], grade_level: str) -> List[Dict[str, str]]:
        """Generates Question-and-Answer flashcards from concepts."""
        flashcards = []
        for c in concepts:
            term = c["term"]
            definition = c["definition"]
            q = f"What is {term}?" if not term.endswith("...") else f"Explain: {term}"
            a = f"{term} is {definition}." if not definition.lower().startswith(term.lower()) else definition
            flashcards.append({
                "grade": grade_level,
                "question": q,
                "answer": a,
                "prompt": f"User: {q}\nAssistant:",
                "completion": f" {a}",
            })
        return flashcards

    def study_book(
        self,
        book_path: Path,
        grade_hint: Optional[str] = None,
        subject: str = "general",
    ) -> Dict[str, Any]:
        """
        Studies an entire book file (PDF, TXT, MD):
        Extracts content, detects grade level, produces human-like study notes,
        generates flashcards, and indexes into the RAG knowledge base.
        """
        p = Path(book_path)
        if not p.is_file():
            raise FileNotFoundError(f"Book file not found: {p.resolve()}")

        raw_text = self.pipeline.extract_text(p)
        if not raw_text.strip():
            raise ValueError(f"Extracted empty content from {p.name}")

        grade_level = self.detect_grade_level(raw_text, hint=grade_hint)
        grade_info = self.GRADE_LEVELS.get(grade_level, {"name": grade_level, "difficulty": 5})

        # Human-like studying: Extract concepts & flashcards
        concepts = self.extract_key_concepts(raw_text)
        flashcards = self.generate_study_flashcards(concepts, grade_level)

        # Save Study Notes & Flashcards
        grade_dir = self.output_dir / grade_level
        grade_dir.mkdir(parents=True, exist_ok=True)

        notes_file = grade_dir / f"{p.stem}_study_notes.md"
        notes_content = [
            f"# Study Notes: {p.stem}",
            f"**Grade Level:** {grade_info['name']} (Level {grade_info['difficulty']})",
            f"**Subject:** {subject.title()}",
            f"**Source Document:** `{p.name}`",
            f"**Key Concepts Identified:** {len(concepts)}",
            "",
            "## 1. Key Concepts & Definitions",
        ]
        for idx, c in enumerate(concepts, 1):
            notes_content.append(f"{idx}. **{c['term']}**: {c['definition']}")

        notes_content.append("\n## 2. Reading Comprehension Flashcards (Q&A)")
        for idx, fc in enumerate(flashcards, 1):
            notes_content.append(f"**Q{idx}:** {fc['question']}\n**A{idx}:** {fc['answer']}\n")

        notes_file.write_text("\n".join(notes_content), encoding="utf-8")

        # Save JSONL flashcards for fine-tuning / self-study
        jsonl_file = grade_dir / f"{p.stem}_flashcards.jsonl"
        with jsonl_file.open("w", encoding="utf-8") as f:
            for fc in flashcards:
                f.write(json.dumps(fc) + "\n")

        # Ingest directly into RAG knowledge base if active
        indexed_chunks = 0
        if self.rag:
            indexed_chunks = self.rag.add_document(notes_file)

        return {
            "book_name": p.name,
            "grade_level": grade_level,
            "grade_name": grade_info["name"],
            "difficulty": grade_info["difficulty"],
            "subject": subject,
            "concepts_learned": len(concepts),
            "flashcards_generated": len(flashcards),
            "study_notes_path": str(notes_file),
            "flashcards_path": str(jsonl_file),
            "rag_chunks_indexed": indexed_chunks,
        }
