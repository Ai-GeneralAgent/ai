from pathlib import Path

files = [
    Path("data_merged_updated.txt"),
    Path("data_merged_expanded.txt"),
    Path(r"D:\book making\llm_knowledge_expansion\datasets\all_text_corpus.txt"),
    Path(r"D:\book making\llm_knowledge_expansion\datasets\books_combined.txt"),
]

patterns = {
    "UTF-8 mojibake (â€)": "\u00e2\u20ac",
    "UTF-8 mojibake (Ã)": "\u00c3",
    "Suspicious Â": "\u00c2",
    "Replacement character": "\ufffd",
}

for path in files:
    print("\n" + "=" * 65)
    print(path)
    print("=" * 65)

    if not path.is_file():
        print("NOT FOUND")
        continue

    print(f"Size: {path.stat().st_size:,} bytes")

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()

    print(f"Characters: {len(text):,}")
    print(f"Source markers: {text.count('--- SOURCE:'):,}")

    for label, pattern in patterns.items():
        print(f"{label}: {text.count(pattern):,}")

    for term in ("Alice", "Alice\u00e2\u20ac\u2122s", "Alice\u2019s"):
        pos = text.find(term)
        if pos >= 0:
            print(f"Sample for {term!r}: {text[max(0, pos-20):pos+100]!r}")
