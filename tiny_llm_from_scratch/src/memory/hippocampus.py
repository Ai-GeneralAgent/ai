"""
Tiered Memory Architecture (AI Hippocampus)
Implements Demis Hassabis' 3-Tier Cognitive Memory:
1. Working Memory (active context buffer)
2. Short-Term Memory (session-level conversation logs in SQLite)
3. Long-Term Durable Memory (semantic/episodic facts, preferences, project knowledge)
Includes automatic fact extraction, Hebbian recall reinforcement, and continual training export.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class DurableFact:
    fact_id: str
    category: str  # 'profile', 'preference', 'project', 'knowledge', 'rule'
    content: str
    confidence: float = 1.0
    source_session: str = "default"
    created_at: float = field(default_factory=time.time)
    last_recalled_at: float = field(default_factory=time.time)
    recall_count: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "category": self.category,
            "content": self.content,
            "confidence": self.confidence,
            "source_session": self.source_session,
            "created_at": self.created_at,
            "last_recalled_at": self.last_recalled_at,
            "recall_count": self.recall_count,
            "tags": self.tags,
        }


class HippocampusMemory:
    """
    Cognitive Hippocampus for Small LLMs:
    Transforms fleeting conversational context into durable, weight-consolidated memory.
    """

    def __init__(self, db_path: Union[str, Path] = "memory/hippocampus.db", working_window: int = 6):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.working_window = working_window
        self.working_memory: List[Dict[str, str]] = []  # In-memory working buffer
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS durable_facts (
                    fact_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL UNIQUE,
                    confidence REAL NOT NULL,
                    source_session TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_recalled_at REAL NOT NULL,
                    recall_count INTEGER DEFAULT 0,
                    tags TEXT
                );

                CREATE TABLE IF NOT EXISTS conversation_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    intent TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_facts_category ON durable_facts(category);
                CREATE INDEX IF NOT EXISTS idx_episodes_session ON conversation_episodes(session_id);
            """)

    # ---------------------------------------------------------
    # Tier 1: Working Memory (Active Context)
    # ---------------------------------------------------------
    def add_to_working_memory(self, role: str, content: str):
        """Maintains the active working memory window."""
        self.working_memory.append({"role": role, "content": content})
        if len(self.working_memory) > self.working_window:
            self.working_memory.pop(0)

    def get_working_context(self) -> List[Dict[str, str]]:
        return list(self.working_memory)

    def clear_working_memory(self):
        self.working_memory.clear()

    # ---------------------------------------------------------
    # Tier 2: Short-Term Memory (Session Episodes)
    # ---------------------------------------------------------
    def log_turn(self, session_id: str, role: str, content: str, intent: Optional[str] = None):
        """Records a turn into the short-term episodic database and working buffer."""
        self.add_to_working_memory(role, content)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversation_episodes (session_id, role, content, timestamp, intent) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, time.time(), intent or "general"),
            )
            conn.commit()

        # If user message, extract facts automatically into Tier 3
        if role == "user":
            self.extract_and_store_facts(content, session_id=session_id)

    def get_recent_episodes(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp, intent FROM conversation_episodes WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    # ---------------------------------------------------------
    # Tier 3: Long-Term Durable Memory (Semantic Facts & Hebbian Recall)
    # ---------------------------------------------------------
    def store_fact(
        self,
        content: str,
        category: str = "knowledge",
        confidence: float = 1.0,
        source_session: str = "default",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Stores or strengthens a long-term fact."""
        clean = content.strip()
        if not clean or len(clean) < 4:
            return False

        fact_id = f"fact_{abs(hash(clean.lower()))}"
        tags_list = tags or self._extract_keywords(clean)
        tags_str = json.dumps(tags_list)

        with self._get_connection() as conn:
            cur = conn.execute("SELECT fact_id, recall_count, confidence FROM durable_facts WHERE content = ?", (clean,))
            existing = cur.fetchone()
            if existing:
                # Strengthen memory through repetition (Hebbian reinforcement)
                new_recall = existing["recall_count"] + 1
                new_conf = min(1.0, existing["confidence"] + 0.05)
                conn.execute(
                    "UPDATE durable_facts SET recall_count = ?, confidence = ?, last_recalled_at = ? WHERE fact_id = ?",
                    (new_recall, new_conf, time.time(), existing["fact_id"]),
                )
            else:
                conn.execute(
                    """INSERT OR IGNORE INTO durable_facts 
                       (fact_id, category, content, confidence, source_session, created_at, last_recalled_at, recall_count, tags)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                    (fact_id, category, clean, confidence, source_session, time.time(), time.time(), tags_str),
                )
            conn.commit()
            return True

    def recall_relevant_memories(self, query: str, top_k: int = 3, min_score: float = 0.1) -> List[DurableFact]:
        """
        Associative recall: searches long-term facts relevant to incoming query.
        Updates recall timestamp and count on retrieved facts.
        """
        query_words = set(self._extract_keywords(query.lower()))
        if not query_words:
            return []

        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM durable_facts ORDER BY confidence DESC, recall_count DESC").fetchall()

            scored = []
            for r in rows:
                content_lower = r["content"].lower()
                tags_list = json.loads(r["tags"] or "[]")
                score = 0.0

                # Match words in content
                content_words = set(self._extract_keywords(content_lower))
                overlap = len(query_words.intersection(content_words))
                if overlap > 0:
                    score += overlap * 1.5

                # Match tags
                tag_overlap = len(query_words.intersection(set(t.lower() for t in tags_list)))
                score += tag_overlap * 2.0

                # Recency & Reinforcement boost
                score += min(1.0, r["recall_count"] * 0.1)
                score *= r["confidence"]

                if score >= min_score:
                    fact = DurableFact(
                        fact_id=r["fact_id"],
                        category=r["category"],
                        content=r["content"],
                        confidence=r["confidence"],
                        source_session=r["source_session"],
                        created_at=r["created_at"],
                        last_recalled_at=r["last_recalled_at"],
                        recall_count=r["recall_count"],
                        tags=tags_list,
                    )
                    scored.append((score, fact))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = [f for _, f in scored[:top_k]]

            # Update recall tracking for accessed facts
            for f in results:
                conn.execute(
                    "UPDATE durable_facts SET recall_count = recall_count + 1, last_recalled_at = ? WHERE fact_id = ?",
                    (time.time(), f.fact_id),
                )
            conn.commit()

            return results

    def format_memory_prompt_prefix(self, query: str) -> str:
        """Constructs memory augmentation block for the model prompt."""
        recalled = self.recall_relevant_memories(query, top_k=3)
        if not recalled:
            return ""

        lines = ["[Durable Memory / AI Hippocampus Recall]"]
        for fact in recalled:
            lines.append(f"- ({fact.category.upper()}) {fact.content}")
        lines.append("")
        return "\n".join(lines)

    # ---------------------------------------------------------
    # Automated Fact Extractor
    # ---------------------------------------------------------
    def extract_and_store_facts(self, text: str, session_id: str = "default") -> List[str]:
        """Detects identity, preferences, decisions, and system rules in conversation turns."""
        extracted = []
        lower = text.strip()

        # 1. Identity & Profile
        name_match = re.search(r"\b(?:my name is|i am|call me)\s+([A-Z][a-zA-Z\s]{1,30})\b", text, re.IGNORECASE)
        if name_match:
            fact = f"User's name is {name_match.group(1).strip()}."
            if self.store_fact(fact, category="profile", source_session=session_id):
                extracted.append(fact)

        # 2. Preferences
        pref_match = re.search(r"\b(?:i prefer|i like|i always want|please always|don't ever|never)\s+([^.?!,;\n]{4,80})", text, re.IGNORECASE)
        if pref_match:
            fact = f"User preference: {pref_match.group(0).strip()}."
            if self.store_fact(fact, category="preference", source_session=session_id):
                extracted.append(fact)

        # 3. Project & Environment Constraints
        if "d drive" in lower or "work in d" in lower:
            fact = "Project environment rule: Always use D: drive, strictly avoid writing to C: drive."
            if self.store_fact(fact, category="rule", source_session=session_id):
                extracted.append(fact)

        if "qwen 3.5" in lower or "qwen3.5" in lower:
            fact = "Model brain preference: Use Qwen 3.5 teacher for reasoning, distillation, and human-like chat."
            if self.store_fact(fact, category="preference", source_session=session_id):
                extracted.append(fact)

        return extracted

    # ---------------------------------------------------------
    # Continual Learning Data Export (for Nightly Sleep Loop)
    # ---------------------------------------------------------
    def export_memories_for_training(self) -> List[Dict[str, str]]:
        """
        Converts confirmed durable memories into instruction-following QA pairs
        suitable for weight-based continual learning consolidation.
        """
        with self._get_connection() as conn:
            rows = conn.execute("SELECT category, content FROM durable_facts WHERE confidence >= 0.8").fetchall()
            training_pairs = []
            for r in rows:
                cat = r["category"]
                content = r["content"]
                if cat == "profile":
                    q = "Who is the user?"
                    a = content
                elif cat == "rule":
                    q = "What are the core project environment rules?"
                    a = content
                elif cat == "preference":
                    q = "What are the user's interaction preferences?"
                    a = content
                else:
                    q = f"What is known about this topic: {content.split()[0]}?"
                    a = content

                training_pairs.append({
                    "instruction": q,
                    "response": a,
                })
            return training_pairs

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "that", "this", "with", "from", "your", "what", "have",
            "are", "was", "not", "but", "can", "all", "any", "were", "when", "there"
        }
        return [w for w in words if w not in stopwords]
