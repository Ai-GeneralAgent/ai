"""Build combined_data.txt from the project's existing data.txt and book files.

Uses data.txt and alice.txt from the project root, plus optional .txt files placed
inside a books/ directory. Backup files in the project root are deliberately ignored.
Original files are never modified.
"""
from pathlib import Path
import argparse
import hashlib
import unicodedata

ROOT = Path(__file__).resolve().parent


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return unicodedata.normalize("NFC", text)


def read_text(path: Path) -> str:
    return normalize_text(path.read_text(encoding="utf-8-sig", errors="replace"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Combine existing data.txt and book text without stripping punctuation.")
    parser.add_argument("--output", default="combined_data.txt", help="Output text file (default: combined_data.txt)")
    parser.add_argument("--data", default="data.txt", help="Base corpus (default: data.txt)")
    parser.add_argument("--alice", default="alice.txt", help="Book file already in your project (default: alice.txt)")
    args = parser.parse_args()

    output_path = (ROOT / args.output).resolve()
    candidate_paths = [ROOT / args.data, ROOT / args.alice]
    books_dir = ROOT / "books"
    if books_dir.is_dir():
        candidate_paths.extend(sorted(p for p in books_dir.rglob("*.txt") if p.is_file()))

    loaded = []
    seen_hashes = set()
    for path in candidate_paths:
        path = path.resolve()
        if path == output_path:
            continue
        if not path.exists() or not path.is_file():
            if path in (ROOT / args.data, ROOT / args.alice):
                print(f"Skipping missing optional file: {path.name}")
            continue
        content = read_text(path)
        if not content.strip():
            print(f"Skipping empty file: {path.name}")
            continue
        digest = sha(content)
        if digest in seen_hashes:
            print(f"Skipping exact duplicate text: {path.name}")
            continue
        seen_hashes.add(digest)
        loaded.append((path, content))

    # If one input file is fully contained in another, do not train on that full
    # text twice. Process the largest texts first so a book already embedded in
    # data.txt is not appended again as alice.txt (or vice versa).
    files = []
    for path, content in sorted(loaded, key=lambda item: len(item[1]), reverse=True):
        if any(content in existing_content for _, existing_content in files):
            print(f"Skipping text already fully contained in a larger source: {path.name}")
            continue
        files.append((path, content))

    if not files:
        raise SystemExit("No text found. Make sure data.txt or alice.txt exists, or put .txt books in books/.")

    combined = "\n\n".join(content.rstrip() for _, content in files) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding="utf-8", newline="\n")

    chars = set(combined)
    symbols = [ch for ch in sorted(chars) if unicodedata.category(ch)[0] in ("P", "S")]
    print("\nIncluded text files:")
    for path, content in files:
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"  {shown}: {len(content):,} characters")
    print(f"\nCreated: {output_path}")
    print(f"Total characters: {len(combined):,}")
    print(f"Unique characters (vocabulary size before keeping old checkpoint characters): {len(chars):,}")
    print(f"Detected punctuation/symbol characters: {len(symbols)}")
    print("Examples: " + " ".join(repr(ch) for ch in symbols[:100]))
    print("\nOriginal data.txt and book files were not changed.")


if __name__ == "__main__":
    main()
