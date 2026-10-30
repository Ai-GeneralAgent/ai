"""
Tiered Memory Architecture (AI Hippocampus)
Implements Demis Hassabis' 3-Tier Cognitive Memory & ChatGPT/Claude Durable Chat Memory:
1. Working Memory (active context buffer & compact rolling dialogue)
2. Short-Term Memory (session-level conversation episodes in SQLite)
3. Long-Term Durable Memory (semantic/episodic facts, user profile, preferences, project constraints)
Includes automatic fact extraction, Hebbian recall reinforcement, context compaction, and continual training export.
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
    category: str  # 'profile', 'preference', 'project', 'knowledge', 'rule', 'identity'
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
    Cognitive Hippocampus for Small & Foundation LLMs:
    Transforms fleeting conversational context into durable, weight-consolidated memory,
    providing ChatGPT and Claude level conversational memory across turns and sessions.
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

    def list_all_facts(self, limit: int = 100) -> List[DurableFact]:
        """Returns all stored durable memories ordered by recency and recall count."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM durable_facts ORDER BY last_recalled_at DESC, recall_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            results = []
            for r in rows:
                results.append(
                    DurableFact(
                        fact_id=r["fact_id"],
                        category=r["category"],
                        content=r["content"],
                        confidence=r["confidence"],
                        source_session=r["source_session"],
                        created_at=r["created_at"],
                        last_recalled_at=r["last_recalled_at"],
                        recall_count=r["recall_count"],
                        tags=json.loads(r["tags"] or "[]"),
                    )
                )
            return results

    def get_core_facts(self, limit: int = 4) -> List[DurableFact]:
        """Returns core user identity, preferences, and project rules that should always be primed."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM durable_facts WHERE category IN ('profile', 'identity', 'rule', 'preference') ORDER BY confidence DESC, recall_count DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                DurableFact(
                    fact_id=r["fact_id"],
                    category=r["category"],
                    content=r["content"],
                    confidence=r["confidence"],
                    source_session=r["source_session"],
                    created_at=r["created_at"],
                    last_recalled_at=r["last_recalled_at"],
                    recall_count=r["recall_count"],
                    tags=json.loads(r["tags"] or "[]"),
                )
                for r in rows
            ]

    def delete_fact(self, target: str) -> int:
        """Deletes a durable memory fact matching fact_id or containing a substring."""
        target_clean = target.strip()
        if not target_clean:
            return 0
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM durable_facts WHERE fact_id = ? OR content LIKE ?",
                (target_clean, f"%{target_clean}%"),
            )
            conn.commit()
            return cur.rowcount

    def clear_all_facts(self) -> int:
        """Clears all durable facts from memory."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM durable_facts")
            conn.commit()
            return cur.rowcount

    def recall_relevant_memories(self, query: str, top_k: int = 3, min_score: float = 0.1) -> List[DurableFact]:
        """
        Associative recall: searches long-term facts relevant to incoming query.
        Updates recall timestamp and count on retrieved facts.
        """
        query_lower = query.lower()
        query_words = set(self._extract_keywords(query_lower))

        # Check if query is asking broadly about identity, preferences, or memory
        broad_memory_query = any(
            phrase in query_lower
            for phrase in [
                "who am i",
                "what is my name",
                "what's my name",
                "do you remember me",
                "what do you remember",
                "tell me about myself",
                "my preferences",
                "what do you know about me",
                "remember anything about me",
                "who created you",
                "who made you",
            ]
        )

        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM durable_facts ORDER BY confidence DESC, recall_count DESC").fetchall()

            scored = []
            for r in rows:
                content_lower = r["content"].lower()
                tags_list = json.loads(r["tags"] or "[]")
                category = r["category"]
                score = 0.0

                # Match words in content
                content_words = set(self._extract_keywords(content_lower))
                overlap = len(query_words.intersection(content_words))
                if overlap > 0:
                    score += overlap * 1.5

                # Match tags
                tag_overlap = len(query_words.intersection(set(t.lower() for t in tags_list)))
                score += tag_overlap * 2.0

                # Broad memory query bonus
                if broad_memory_query:
                    if category in ["profile", "identity"]:
                        score += 3.0
                    elif category in ["preference", "rule"]:
                        score += 1.5

                # Recency & Reinforcement boost
                score += min(1.0, r["recall_count"] * 0.1)
                score *= r["confidence"]

                if score >= min_score or (broad_memory_query and category in ["profile", "identity"]):
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

    def format_system_memory_context(self, query: str = "", top_k: int = 5) -> str:
        """
        Constructs Claude/ChatGPT-style structured memory block for injection into system persona.
        Combines persistent core profile facts with dynamically retrieved episodic facts.
        """
        core_facts = self.get_core_facts(limit=3)
        recalled = self.recall_relevant_memories(query, top_k=top_k) if query else []

        # Merge deduplicated by fact_id
        seen = set()
        merged: List[DurableFact] = []
        for f in core_facts + recalled:
            if f.fact_id not in seen:
                seen.add(f.fact_id)
                merged.append(f)

        if not merged:
            return ""

        lines = [
            "<durable_user_memory>",
            "The following user and project memories have been recalled from previous interactions:",
        ]
        for f in merged:
            lines.append(f"- [{f.category.upper()}] {f.content}")
        lines.append("Use these memories naturally to personalize your response without repeating the list back to the user unless asked.")
        lines.append("</durable_user_memory>")
        return "\n".join(lines)

    # ---------------------------------------------------------
    # Automated Fact Extractor (Claude & ChatGPT Style)
    # ---------------------------------------------------------
    def extract_and_store_facts(self, text: str, session_id: str = "default") -> List[str]:
        """Detects identity, preferences, decisions, and system rules in conversation turns."""
        extracted = []
        lower = text.strip().lower()

        # 0. Explicit Memory Directives (e.g. "Remember that...", "Please remember...", "Note that...")
        directive_match = re.search(
            r"\b(?:remember\s+that|please\s+remember\s+that|remember\s+this|remember|please\s+remember|keep\s+in\s+mind\s+that|keep\s+in\s+mind|note\s+that|don't\s+forget\s+that)\s+([^.?!;\n]{3,120})",
            text,
            re.IGNORECASE,
        )
        if directive_match:
            raw_fact = directive_match.group(1).strip()
            # Normalize first-person pronouns
            norm_fact = re.sub(r"\b(?:i\s+am|i'm)\b", "User is", raw_fact, flags=re.IGNORECASE)
            norm_fact = re.sub(r"\b(?:my)\b", "User's", norm_fact, flags=re.IGNORECASE)
            norm_fact = re.sub(r"\b(?:i\s+have)\b", "User has", norm_fact, flags=re.IGNORECASE)
            norm_fact = re.sub(r"\b(?:i\s+like|i\s+love|i\s+prefer)\b", "User prefers", norm_fact, flags=re.IGNORECASE)
            norm_fact = re.sub(r"\b(?:we\s+are)\b", "Project is", norm_fact, flags=re.IGNORECASE)
            norm_fact = re.sub(r"\b(?:our)\b", "Project's", norm_fact, flags=re.IGNORECASE)
            if not norm_fact.endswith("."):
                norm_fact += "."
            fact = f"User noted: {norm_fact}"
            if self.store_fact(fact, category="preference", source_session=session_id):
                extracted.append(fact)

        # 1. Identity & Profile
        name_match = re.search(r"\b(?:my name is|i am|call me)\s+([A-Z][a-zA-Z\s]{1,30})\b", text, re.IGNORECASE)
        if name_match:
            cand = name_match.group(1).strip()
            # Filter out non-names like "tired", "fine", "ready"
            if cand.lower() not in ["fine", "good", "ready", "happy", "tired", "here", "curious", "interested"]:
                fact = f"User's name is {cand}."
                if self.store_fact(fact, category="profile", source_session=session_id):
                    extracted.append(fact)

        job_match = re.search(r"\b(?:i work as|i am an?|my profession is|my role is|my job is)\s+([a-zA-Z0-9_\s]{3,40})\b", text, re.IGNORECASE)
        if job_match:
            job = job_match.group(1).strip()
            if job.lower() not in ["thinking", "ready", "here", "fine", "looking"]:
                fact = f"User works as {job}."
                if self.store_fact(fact, category="profile", source_session=session_id):
                    extracted.append(fact)

        loc_match = re.search(r"\b(?:i live in|i'm based in|i am from|my location is)\s+([A-Za-z0-9_\s,]{3,40})\b", text, re.IGNORECASE)
        if loc_match:
            loc = loc_match.group(1).strip()
            fact = f"User is located in {loc}."
            if self.store_fact(fact, category="profile", source_session=session_id):
                extracted.append(fact)

        # 2. Preferences & Favorites
        fav_match = re.search(r"\b(?:my favorite|my preferred)\s+([a-zA-Z0-9_\s]{2,25})\s+is\s+([^.?!,;\n]{2,50})", text, re.IGNORECASE)
        if fav_match:
            item = fav_match.group(1).strip()
            val = fav_match.group(2).strip()
            fact = f"User's favorite {item} is {val}."
            if self.store_fact(fact, category="preference", source_session=session_id):
                extracted.append(fact)

        pref_match = re.search(r"\b(?:i prefer|i like|i always want|please always|don't ever|never)\s+([^.?!,;\n]{4,80})", text, re.IGNORECASE)
        if pref_match:
            fact = f"User preference: {pref_match.group(0).strip()}."
            if self.store_fact(fact, category="preference", source_session=session_id):
                extracted.append(fact)

        # 3. Project & Environment Constraints
        proj_match = re.search(r"\b(?:our project is|the project is|we are building|my project is)\s+([^.?!,;\n]{3,80})", text, re.IGNORECASE)
        if proj_match:
            p_desc = proj_match.group(1).strip()
            fact = f"Project focus: {p_desc}."
            if self.store_fact(fact, category="project", source_session=session_id):
                extracted.append(fact)

        if "d drive" in lower or "work in d" in lower:
            fact = "Project environment rule: Always use D: drive, strictly avoid writing to C: drive."
            if self.store_fact(fact, category="rule", source_session=session_id):
                extracted.append(fact)

        # 4. Identity & Creator Recognition
        if any(w in lower for w in ["who created you", "who made you", "yuvraj", "thakur", "your identity"]):
            fact = "Model identity: LLM was created and developed by Yuvraj Thakur."
            if self.store_fact(fact, category="identity", source_session=session_id):
                extracted.append(fact)

        return extracted

    # ---------------------------------------------------------
    # Context Compaction & Rolling Summary (ChatGPT/Claude Style)
    # ---------------------------------------------------------
    @staticmethod
    def compact_conversation_history(
        messages: List[Dict[str, Any]], keep_recent: int = 8
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Implements ChatGPT & Claude conversational context compaction:
        When message history exceeds keep_recent turns, compresses earlier dialogue turns
        into a concise summary context note while preserving recent turns intact.
        """
        if len(messages) <= keep_recent:
            return messages, None

        older = messages[:-keep_recent]
        recent = messages[-keep_recent:]

        topics = []
        for msg in older:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue
            first_line = content.splitlines()[0][:70]
            if role == "user":
                topics.append(f"User inquired: '{first_line}'")
            elif role == "assistant":
                topics.append(f"Assistant resolved: '{first_line}'")

        summary_body = " | ".join(topics[-6:])  # Last few compressed points
        summary_note = f"[Prior Conversation Context Summary ({len(older)} earlier turns): {summary_body}]"

        return recent, summary_note

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
