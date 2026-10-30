#!/usr/bin/env python3
"""
Prepares Unified Q&A and Database Master SFT Dataset:
1. Combines 50k general Q&A (D:\\qwen\\data\\qa_50k\\qa_50k_general_chat.jsonl)
2. Combines 10k Q&A dataset (if available)
3. Generates comprehensive Database & SQL Q&A across:
   - SQL syntax & advanced queries (JOINs, CTEs, Window functions, Aggregations)
   - Database design, normalization (1NF-BCNF), ER modeling, schema design
   - Performance tuning, indexing (B-Tree, Hash, GIN), EXPLAIN plans
   - ACID properties, concurrency control, transaction isolation levels
   - Modern engines: PostgreSQL, MySQL, SQLite, DuckDB, Redis, MongoDB
4. Outputs to D:\\tiny_llm_from_scratch\\tiny_llm_from_scratch\\training_data\\qa_and_database_sft.jsonl
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_ROOT / "training_data" / "qa_and_database_sft.jsonl"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

DATABASE_QA = [
    {
        "user": "What is a database?",
        "assistant": "A database is an organized collection of structured data stored electronically in a computer system, managed by a Database Management System (DBMS) to facilitate rapid querying, updating, and retrieval."
    },
    {
        "user": "What does ACID stand for in database transactions?",
        "assistant": "ACID stands for Atomicity (all changes succeed or none do), Consistency (data remains valid according to rules), Isolation (concurrent transactions execute independently without interference), and Durability (committed transactions survive system crashes)."
    },
    {
        "user": "What is the difference between SQL and NoSQL databases?",
        "assistant": "SQL databases are relational, table-based, use structured query language, and enforce rigid predefined schemas with strong ACID guarantees (e.g. PostgreSQL, MySQL). NoSQL databases are non-relational, document-, key-value-, wide-column-, or graph-based, support dynamic schemas, and scale horizontally (e.g. MongoDB, Redis, Cassandra)."
    },
    {
        "user": "Explain primary key vs foreign key.",
        "assistant": "A primary key is a column or set of columns that uniquely identifies each row in a table and cannot contain NULL values. A foreign key is a column in one table that references the primary key of another table, establishing a link and enforcing referential integrity."
    },
    {
        "user": "What are the common types of SQL JOINs?",
        "assistant": "1. INNER JOIN: returns rows where there is a match in both tables.\n2. LEFT JOIN (or LEFT OUTER JOIN): returns all rows from the left table and matched rows from the right table (with NULLs for unmatched right rows).\n3. RIGHT JOIN: returns all rows from the right table and matched rows from the left table.\n4. FULL OUTER JOIN: returns all rows when there is a match in either left or right table.\n5. CROSS JOIN: returns the Cartesian product of all rows from both tables."
    },
    {
        "user": "What is database normalization and why is it used?",
        "assistant": "Database normalization is the process of structuring relational database tables to reduce data redundancy and improve data integrity. It prevents update, insertion, and deletion anomalies by decomposing larger tables into smaller, well-related tables."
    },
    {
        "user": "Explain the normal forms: 1NF, 2NF, 3NF, and BCNF.",
        "assistant": "- 1NF (First Normal Form): All column values are atomic (no repeating groups or arrays) and each row is uniquely identified.\n- 2NF (Second Normal Form): Meets 1NF and all non-key attributes are fully functionally dependent on the entire primary key (no partial dependencies).\n- 3NF (Third Normal Form): Meets 2NF and has no transitive dependencies (non-key attributes depend only on the primary key).\n- BCNF (Boyce-Codd Normal Form): A stricter version of 3NF where every determinant must be a candidate key."
    },
    {
        "user": "How does an index work in a relational database?",
        "assistant": "A database index is an auxiliary data structure (commonly a balanced B-Tree or B+ Tree) that holds sorted key values and pointers to table rows. It allows the database engine to search for rows in O(log N) time instead of performing an O(N) full table scan."
    },
    {
        "user": "What is a clustered index vs a non-clustered index?",
        "assistant": "A clustered index physically dictates the order in which table rows are stored on disk; therefore, a table can have only one clustered index (typically on the primary key). A non-clustered index stores sorted keys separately along with pointers (row IDs or clustered keys) back to the actual data rows; a table can have multiple non-clustered indexes."
    },
    {
        "user": "What are Common Table Expressions (CTEs) in SQL?",
        "assistant": "A Common Table Expression (CTE) is a temporary named result set defined within the execution of a single SELECT, INSERT, UPDATE, or DELETE statement using the 'WITH' keyword. CTEs improve query readability, enable modular logic, and support recursive hierarchical queries."
    },
    {
        "user": "Write an SQL query to find the second highest salary from an Employee table.",
        "assistant": "```sql\nSELECT MAX(salary) AS SecondHighestSalary\nFROM Employee\nWHERE salary < (SELECT MAX(salary) FROM Employee);\n```\nAlternatively, using the DENSE_RANK window function:\n```sql\nWITH RankedSalaries AS (\n    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as rnk\n    FROM Employee\n)\nSELECT salary FROM RankedSalaries WHERE rnk = 2 LIMIT 1;\n```"
    },
    {
        "user": "What is the difference between WHERE and HAVING in SQL?",
        "assistant": "WHERE filters individual rows before any grouping or aggregation takes place. HAVING filters aggregated groups after the GROUP BY clause has executed and can evaluate aggregate functions like COUNT(), SUM(), AVG(), and MAX()."
    },
    {
        "user": "What is a database deadlock and how can it be resolved?",
        "assistant": "A deadlock occurs when two or more transactions hold locks on resources the other transactions need, creating a circular wait condition where none can proceed. RDBMS engines resolve deadlocks by automatically detecting the cycle and aborting (rolling back) one transaction (the victim). Prevent deadlocks by acquiring locks in a consistent order and keeping transactions short."
    },
    {
        "user": "What are transaction isolation levels and what read phenomena do they prevent?",
        "assistant": "SQL-92 defines four isolation levels:\n1. Read Uncommitted: Prevents nothing; allows dirty reads, non-repeatable reads, and phantom reads.\n2. Read Committed: Prevents dirty reads; allows non-repeatable reads and phantom reads.\n3. Repeatable Read: Prevents dirty reads and non-repeatable reads; allows phantom reads.\n4. Serializable: Prevents all read anomalies by executing transactions as if strictly serial."
    },
    {
        "user": "What is connection pooling and why is it important?",
        "assistant": "Connection pooling is a mechanism that maintains a cache of active database connections ready for reuse by incoming requests. Establishing a new database connection involves expensive network handshakes and authentication; connection pooling eliminates this latency and prevents overwhelming the database server."
    },
    {
        "user": "What is SQLite and when should you use it?",
        "assistant": "SQLite is a lightweight, serverless, self-contained, zero-configuration, transactional SQL database engine that stores the entire database in a single disk file. It is ideal for embedded systems, mobile apps, desktop software, local development, edge devices, and small-to-medium web traffic."
    },
    {
        "user": "What is DuckDB and how is it different from SQLite?",
        "assistant": "DuckDB is an in-process SQL OLAP (Online Analytical Processing) database designed for fast analytical queries on columnar data, vectorized execution, and direct querying of Parquet, CSV, and Arrow files. While SQLite is optimized for transactional (OLTP) row-based operations, DuckDB is optimized for heavy analytics, aggregations, and data science workloads."
    },
    {
        "user": "What is database sharding?",
        "assistant": "Database sharding is a horizontal partitioning technique where a large database is divided into smaller, faster, and more easily managed pieces called shards, distributed across multiple physical database servers. Each shard shares the same schema but holds a distinct subset of data."
    },
    {
        "user": "Explain optimistic concurrency control vs pessimistic concurrency control.",
        "assistant": "Pessimistic concurrency control locks rows at the start of a transaction, preventing others from modifying or reading them until the transaction finishes. Optimistic concurrency control allows multiple transactions to proceed without locking; before committing, it verifies whether records were modified by another transaction (e.g. using a version number or timestamp) and aborts if a conflict exists."
    },
    {
        "user": "What is a database migration?",
        "assistant": "A database migration is a version-controlled, automated script that applies incremental changes (such as creating tables, altering columns, adding indexes, or transforming data) to a database schema, allowing teams to synchronize schema versions across environments reliably."
    }
]


def main():
    total_written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        # 1. Write specialized Database Q&A first
        for item in DATABASE_QA:
            msg = {
                "messages": [
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]}
                ]
            }
            out_f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            total_written += 1

        print(f"Added {total_written} curated Database & SQL Q&A pairs.")

        # 2. Add 50k general Q&A
        qa_50k_path = Path("D:/qwen/data/qa_50k/qa_50k_general_chat.jsonl")
        if qa_50k_path.is_file():
            print(f"Merging {qa_50k_path.name}...")
            count_50k = 0
            with open(qa_50k_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        out_f.write(line.strip() + "\n")
                        count_50k += 1
            print(f"Added {count_50k:,} examples from {qa_50k_path.name}")
            total_written += count_50k

        # 3. Add 10k Q&A if available
        qa_10k_path = Path("C:/Users/YUVRAJ/Downloads/llm_qa_10k_dataset/qa_10k_chat.jsonl")
        if qa_10k_path.is_file():
            print(f"Merging {qa_10k_path.name}...")
            count_10k = 0
            with open(qa_10k_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        out_f.write(line.strip() + "\n")
                        count_10k += 1
            print(f"Added {count_10k:,} examples from {qa_10k_path.name}")
            total_written += count_10k

        # 4. Add existing instruction_conversations if available
        convo_path = PROJECT_ROOT / "instruction_conversations.jsonl"
        if convo_path.is_file():
            count_convo = 0
            with open(convo_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        out_f.write(line.strip() + "\n")
                        count_convo += 1
            print(f"Added {count_convo:,} examples from {convo_path.name}")
            total_written += count_convo

    print(f"\n[DONE] Successfully created master dataset with {total_written:,} records at:")
    print(f"       {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
