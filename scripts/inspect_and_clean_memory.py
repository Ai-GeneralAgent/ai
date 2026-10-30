#!/usr/bin/env python3
"""
Inspects and purges any Qwen, Alibaba, or third-party creator references from
all memory databases (SQLite), and establishes the canonical durable identity:
- Model Name: LLM
- Creator: Yuvraj Thakur
- Nature: Friendly, honest, and helpful conversational AI
"""
import glob
import os
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_FILES = [
    PROJECT_ROOT / "memory.db",
    PROJECT_ROOT / "memory" / "hippocampus.db",
    PROJECT_ROOT / "tiny_llm_from_scratch" / "memory.db",
]

FORBIDDEN_WORDS = ["qwen", "alibaba", "tongyi", "dashscope"]

print("=" * 70)
print("INSPECTING AND SANITIZING MEMORY DATABASES")
print("=" * 70)

for db_path in DB_FILES:
    if not db_path.is_file():
        print(f"[SKIP] DB not found: {db_path}")
        continue

    print(f"\n[DATABASE] Processing: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    print(f"  Found tables: {tables}")

    purged_rows = 0

    for table in tables:
        # Get column names
        cur.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cur.fetchall()]
        cur.execute(f"SELECT rowid, * FROM {table}")
        rows = cur.fetchall()

        for row in rows:
            rowid = row[0]
            data_cols = row[1:]
            row_str = " ".join(str(c) for c in data_cols).lower()

            if any(fw in row_str for fw in FORBIDDEN_WORDS):
                print(f"  [PURGING] Table '{table}', rowid {rowid}: {str(data_cols)[:120]}")
                cur.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
                purged_rows += 1

    # Ensure durable facts reflect correct identity
    if "durable_facts" in tables or "durable_memory" in tables:
        t_name = "durable_facts" if "durable_facts" in tables else "durable_memory"
        cur.execute(f"PRAGMA table_info({t_name})")
        cols = [c[1] for c in cur.fetchall()]

        # Clear any old identity records
        if "category" in cols:
            cur.execute(f"DELETE FROM {t_name} WHERE category = 'identity'")
        elif "key" in cols:
            cur.execute(f"DELETE FROM {t_name} WHERE key LIKE '%identity%' OR key LIKE '%creator%'")

        # Insert canonical identity
        if "fact" in cols and "category" in cols:
            cur.execute(
                f"INSERT INTO {t_name} (category, fact, confidence, source) VALUES (?, ?, ?, ?)",
                ("identity", "The model is LLM, created and built by Yuvraj Thakur.", 1.0, "system_core"),
            )
            cur.execute(
                f"INSERT INTO {t_name} (category, fact, confidence, source) VALUES (?, ?, ?, ?)",
                ("identity", "The model is not created by Alibaba, OpenAI, or Google; it is an independent LLM developed by Yuvraj Thakur.", 1.0, "system_core"),
            )
        print(f"  [IDENTITY] Inserted canonical durable facts into '{t_name}'.")

    conn.commit()
    conn.close()
    print(f"  [OK] Cleaned {db_path.name}: {purged_rows} forbidden references purged.")

print("\n" + "=" * 70)
print("MEMORY SANITIZATION COMPLETE")
print("=" * 70)

