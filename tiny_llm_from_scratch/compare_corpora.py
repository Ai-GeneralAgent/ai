from pathlib import Path
import hashlib

folder = Path(r"D:\book making\llm_knowledge_expansion\datasets")
a_path = folder / "all_text_corpus.txt"
b_path = folder / "books_combined.txt"

for path in (a_path, b_path):
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")

a = a_path.read_text(encoding="utf-8-sig", errors="replace")
b = b_path.read_text(encoding="utf-8-sig", errors="replace")

print("all_text_corpus characters:", f"{len(a):,}")
print("books_combined characters:", f"{len(b):,}")
print("Exactly identical:", a == b)
print("Identical SHA-256:", hashlib.sha256(a.encode()).hexdigest()
      == hashlib.sha256(b.encode()).hexdigest())

# Compare normalized lines as a rough estimate of repeated content.
lines_a = {line.strip() for line in a.splitlines() if len(line.strip()) > 50}
lines_b = {line.strip() for line in b.splitlines() if len(line.strip()) > 50}
shared = lines_a & lines_b

print("Distinct long lines in all_text_corpus:", f"{len(lines_a):,}")
print("Distinct long lines in books_combined:", f"{len(lines_b):,}")
print("Shared long lines:", f"{len(shared):,}")
print(
    "Share of books_combined long lines also found in all_text_corpus:",
    f"{100 * len(shared) / max(1, len(lines_b)):.1f}%"
)
