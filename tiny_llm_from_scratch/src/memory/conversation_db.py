import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Union


class ConversationDatabase:
    """SQLite-backed persistent conversation memory and user preference storage."""

    def __init__(self, db_path: Union[str, Path] = "memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    summary TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS project_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)

    def create_session(self, session_id: str, title: str = "New Session") -> str:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, title, created_at) VALUES (?, ?, ?)",
                (session_id, title, time.time()),
            )
        return session_id

    def add_message(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, time.time(), json.dumps(metadata or {})),
            )

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT role, content, timestamp, metadata FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "metadata": json.loads(r["metadata"] or "{}"),
                }
                for r in rows
            ]

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT session_id, title, created_at, summary FROM sessions ORDER BY created_at DESC")
            return [dict(r) for r in cur.fetchall()]

    def search_history(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT session_id, role, content, timestamp FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", limit),
            )
            return [dict(r) for r in cur.fetchall()]

    def set_preference(self, key: str, value: str):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )

    def get_preferences(self) -> Dict[str, str]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT key, value FROM preferences")
            return {r["key"]: r["value"] for r in cur.fetchall()}

    def clear_memory(self):
        with self._get_connection() as conn:
            conn.executescript("""
                DELETE FROM messages;
                DELETE FROM sessions;
                DELETE FROM project_notes;
            """)
