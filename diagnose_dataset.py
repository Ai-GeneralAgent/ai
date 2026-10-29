from pathlib import Path
import hashlib

root = Path(r"D:\book making\llm_knowledge_expansion\datasets")
merged = Path("data_merged_expanded.txt")

patterns = ["â€", "â€™", "Ã", "Â", "ðŸ", "\ufffd"]

print("=" * 65)
print("LLM DATASET DIAGNOSTIC")
print("=" * 65)

if not root.is_dir():
    raise SystemExit(f"Dataset folder not found: {root}")

files = sorted(
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() == ".txt"
)

print(f"\nSource TXT files: {len(files)}")
print("\nSOURCE FILE REPORT")
print("-" * 65)

hashes = {}
total_chars = 0
total_mojibake = 0

for path in files:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
    except OSError as exc:
        print(f"READ ERROR: {path.name}: {exc}")
        continue

    total_chars += len(text)
    hits = sum(text.count(p) for p in patterns)
    total_mojibake += hits
    digest = hashlib.sha256(raw).hexdigest()
    hashes.setdefault(digest, []).append(path.name)

    print(
        f"{path.name[:42]:42} "
        f"{len(text):>10,} chars  "
        f"encoding-warning-patterns: {hits}"
    )

duplicates = [names for names in hashes.values() if len(names) > 1]

print("\nSUMMARY")
print("-" * 65)
print(f"Total source characters: {total_chars:,}")
print(f"Encoding warning-pattern occurrences: {total_mojibake}")
print(f"Exact duplicate groups: {len(duplicates)}")

for group in duplicates[:10]:
    print("Duplicate:", " | ".join(group))

if merged.is_file():
    print("\nMERGED DATASET")
    print("-" * 65)
    print(f"Path: {merged.resolve()}")
    print(f"Size: {merged.stat().st_size:,} bytes")

    data = merged.read_text(encoding="utf-8-sig", errors="replace")
    print(f"Characters: {len(data):,}")
    print(f"Source markers: {data.count('--- SOURCE:'):,}")
    print(
        "Encoding warning-pattern occurrences:",
        sum(data.count(p) for p in patterns)
    )

    print("\nFIRST SOURCE FILENAMES")
    for path in files[:15]:
        print(" -", path.name)
else:
    print(f"\nExpanded dataset not found: {merged.resolve()}")
