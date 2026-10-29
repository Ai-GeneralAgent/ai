"""
Merge existing TXT datasets and optional PDF/TXT books into data_merged.txt.

Default root-level inputs:
  alice.txt
  data_backup_before_expansion.txt
  data_backup.txt
  data_before_expanded.txt
  data.txt

Optional extra files can be placed in books/, pdfs/, or texts/.
PDF support extracts selectable text using pypdf. Scanned/image-only PDFs need OCR.
This ZIP contains only this Python script, not any books or PDF documents.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

DEFAULT_SOURCE_FILES = [
    "alice.txt",
    "data_backup_before_expansion.txt",
    "data_backup.txt",
    "data_before_expanded.txt",
    "data.txt",
]
SEARCH_FOLDERS = ("books", "pdfs", "texts")


def read_txt(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs pypdf. Install it with: python -m pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    page_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", "").strip()
        if text:
            page_text.append(f"[Page {page_num}]\n{text}")
    result = "\n\n".join(page_text)
    if not result:
        raise ValueError(
            "No selectable text found. This may be a scanned PDF; OCR it first."
        )
    return result


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)

    # Remove common Project Gutenberg header/footer when present.
    low = text.lower()
    start_markers = (
        "*** start of the project gutenberg",
        "*** start of this project gutenberg",
    )
    start_positions = [low.find(x) for x in start_markers if low.find(x) >= 0]
    if start_positions:
        pos = min(start_positions)
        line_end = text.find("\n", pos)
        if line_end >= 0:
            text = text[line_end + 1:]

    low = text.lower()
    end_markers = (
        "*** end of the project gutenberg",
        "*** end of this project gutenberg",
    )
    end_positions = [low.find(x) for x in end_markers if low.find(x) >= 0]
    if end_positions:
        text = text[:min(end_positions)]

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_paragraphs(text: str):
    for part in re.split(r"\n\s*\n", text):
        # Join line-wrapped PDF/TXT paragraphs, preserving paragraph boundaries.
        paragraph = re.sub(r"\s*\n\s*", " ", part).strip()
        paragraph = re.sub(r"[ \t]{2,}", " ", paragraph)
        if paragraph:
            yield paragraph


def discover_inputs(root: Path, root_files: list[str], output_path: Path) -> list[Path]:
    found = []
    seen_paths = set()

    for filename in root_files:
        path = root / filename
        if path.is_file() and path.resolve() != output_path and path.resolve() not in seen_paths:
            found.append(path)
            seen_paths.add(path.resolve())
        elif not path.exists():
            print(f"[SKIP] Not found: {filename}")

    for folder_name in SEARCH_FOLDERS:
        folder = root / folder_name
        if folder.is_dir():
            candidates = []
            for pattern in ("*.txt", "*.TXT", "*.pdf", "*.PDF"):
                candidates.extend(folder.rglob(pattern))
            for path in sorted(candidates):
                resolved = path.resolve()
                if resolved != output_path and resolved not in seen_paths:
                    found.append(path)
                    seen_paths.add(resolved)

    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the previous TinyGPT data files plus optional TXT/PDF books."
    )
    parser.add_argument(
        "--output",
        default="data_merged.txt",
        help="New output filename (default: data_merged.txt)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_SOURCE_FILES,
        help="Root-level source filenames; defaults to your five listed data files.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    output_path = (root / args.output).resolve()
    files = discover_inputs(root, args.files, output_path)

    if not files:
        print("No input files found. Run this from your tiny_llm_from_scratch folder.")
        sys.exit(1)

    unique_paragraphs = []
    seen_paragraphs = set()
    skipped_duplicates = 0
    files_ok = 0
    raw_chars = 0

    for path in files:
        try:
            raw = read_pdf(path) if path.suffix.lower() == ".pdf" else read_txt(path)
            raw_chars += len(raw)
            text = clean_text(raw)
            before = len(unique_paragraphs)

            for paragraph in iter_paragraphs(text):
                # Exact, case-sensitive deduplication; similar-but-not-identical
                # paragraphs are retained to avoid deleting meaningful variants.
                if paragraph in seen_paragraphs:
                    skipped_duplicates += 1
                    continue
                seen_paragraphs.add(paragraph)
                unique_paragraphs.append(paragraph)

            added = len(unique_paragraphs) - before
            files_ok += 1
            print(f"[OK] {path}: {len(raw):,} characters; {added:,} unique paragraphs added")
        except Exception as exc:
            print(f"[WARNING] Could not process {path}: {exc}", file=sys.stderr)

    if not unique_paragraphs:
        print("No text could be extracted from the input files.", file=sys.stderr)
        sys.exit(2)

    merged = "\n\n".join(unique_paragraphs) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(merged, encoding="utf-8", newline="\n")

    symbols = sorted({c for c in merged if not c.isalnum() and not c.isspace()})
    print("\n--- Merge report ---")
    print(f"Files processed: {files_ok}")
    print(f"Source characters read: {raw_chars:,}")
    print(f"Duplicate paragraphs skipped: {skipped_duplicates:,}")
    print(f"Output characters: {len(merged):,}")
    print(f"Unique characters: {len(set(merged)):,}")
    print(f"Symbols/punctuation found: {''.join(symbols)}")
    print(f"Saved output: {output_path}")
    print("\nThis created a merged corpus; it did not change your model checkpoint.")
    print("For PDF files, selectable text is extracted. Scanned PDFs require OCR.")


if __name__ == "__main__":
    main()
