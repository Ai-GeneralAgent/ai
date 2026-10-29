"""
Merge TXT/PDF books into a training corpus for TinyGPT.

- Reads UTF-8 TXT files.
- Extracts selectable text from PDF files with pypdf.
- Includes known local datasets/backups when present.
- Removes common Project Gutenberg boilerplate and exact duplicate paragraphs.
- Writes a NEW output file; never overwrites source files.

Important:
PDFs that contain scanned page images need OCR first. pypdf extracts embedded/selectable
text; it does not perform OCR or guarantee identical formatting to the original TXT.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


DEFAULT_SOURCE_FILES = [
    "data.txt",
    "data_backup.txt",
    "data_backup_before_expansion.txt",
    "alice.txt",
]

DEFAULT_INPUT_DIRS = ["books", "pdfs", "texts"]


def read_txt(path: Path) -> str:
    """Read text using common encodings, preferring UTF-8."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    """Extract selectable text from each PDF page, retaining page boundaries."""
    if PdfReader is None:
        raise RuntimeError(
            "PDF support requires pypdf. Install it with:\n"
            "  python -m pip install pypdf"
        )
    reader = PdfReader(str(path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\x00", "")
        if text.strip():
            pages.append(f"\n[Page {page_num}]\n{text.strip()}")
    result = "\n\n".join(pages)
    if not result.strip():
        raise ValueError(
            "No selectable text was extracted. This may be a scanned/image-only PDF; "
            "run OCR on it first, then save/export the recognized text."
        )
    return result


def normalize_text(text: str) -> str:
    """Normalize Unicode and line endings without stripping useful punctuation."""
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    # Remove common Gutenberg start/end boilerplate, when clearly marked.
    start_markers = (
        "*** start of the project gutenberg",
        "*** start of this project gutenberg",
    )
    lower = text.lower()
    starts = [lower.find(marker) for marker in start_markers if lower.find(marker) >= 0]
    if starts:
        pos = min(starts)
        endline = text.find("\n", pos)
        if endline >= 0:
            text = text[endline + 1:]

    end_markers = (
        "*** end of the project gutenberg",
        "*** end of this project gutenberg",
        "end of the project gutenberg",
    )
    lower = text.lower()
    positions = [lower.find(marker) for marker in end_markers if lower.find(marker) >= 0]
    if positions:
        text = text[:min(positions)]

    # Normalize excessive whitespace but preserve paragraph boundaries.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def paragraphs(text: str):
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s*\n\s*", " ", para).strip()
        para = re.sub(r"[ \t]{2,}", " ", para)
        if para:
            yield para


def discover_files(root: Path, explicit: list[str]) -> list[Path]:
    found = []
    seen = set()

    for name in explicit:
        p = root / name
        if p.is_file():
            resolved = p.resolve()
            if resolved not in seen:
                found.append(p)
                seen.add(resolved)

    for folder_name in DEFAULT_INPUT_DIRS:
        folder = root / folder_name
        if folder.is_dir():
            for pattern in ("*.txt", "*.TXT", "*.pdf", "*.PDF"):
                for p in sorted(folder.rglob(pattern)):
                    resolved = p.resolve()
                    if resolved not in seen:
                        found.append(p)
                        seen.add(resolved)

    return found


def main():
    parser = argparse.ArgumentParser(
        description="Merge local TXT and PDF books into a clean TinyGPT training corpus."
    )
    parser.add_argument(
        "--output", default="data_merged.txt",
        help="Output text file (default: data_merged.txt)"
    )
    parser.add_argument(
        "--files", nargs="*",
        default=DEFAULT_SOURCE_FILES,
        help="Additional source filenames in the project directory"
    )
    parser.add_argument(
        "--include-output", action="store_true",
        help="Include existing data_merged.txt if present (normally avoided)"
    )
    args = parser.parse_args()

    root = Path.cwd()
    output_path = (root / args.output).resolve()
    source_names = list(args.files)
    if args.include_output and Path("data_merged.txt").exists():
        source_names.append("data_merged.txt")

    files = discover_files(root, source_names)
    # Never read the output we're about to create as an input by accident.
    files = [p for p in files if p.resolve() != output_path]

    if not files:
        print("No input files found. Put TXT/PDF files in books/, pdfs/, or texts/,")
        print("or place data.txt/alice.txt in the current project folder.")
        sys.exit(1)

    seen_paragraphs = set()
    merged_paragraphs = []
    report = []
    total_chars = 0
    skipped_duplicates = 0

    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                raw = read_pdf(path)
            elif path.suffix.lower() == ".txt":
                raw = read_txt(path)
            else:
                continue

            cleaned = normalize_text(raw)
            count_before = len(merged_paragraphs)
            added_chars = 0

            for para in paragraphs(cleaned):
                # Case-sensitive dedupe: don't merge differently capitalized content.
                key = para
                if key in seen_paragraphs:
                    skipped_duplicates += 1
                    continue
                seen_paragraphs.add(key)
                merged_paragraphs.append(para)
                added_chars += len(para)

            report.append(
                (str(path.relative_to(root)), len(raw), len(merged_paragraphs) - count_before, added_chars)
            )
            total_chars += len(raw)
            print(
                f"[OK] {path}: {len(raw):,} source characters; "
                f"{len(merged_paragraphs) - count_before:,} new paragraphs"
            )
        except Exception as exc:
            print(f"[WARNING] Could not read {path}: {exc}", file=sys.stderr)

    if not merged_paragraphs:
        print("No text could be extracted. Check the warnings above.", file=sys.stderr)
        sys.exit(2)

    output_text = "\n\n".join(merged_paragraphs) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8", newline="\n")

    symbols = sorted({
        ch for ch in output_text
        if not ch.isalnum() and not ch.isspace()
    })
    print("\n--- Merge report ---")
    print(f"Files processed: {len(report)}")
    print(f"Source characters read: {total_chars:,}")
    print(f"Duplicate paragraphs skipped: {skipped_duplicates:,}")
    print(f"Output characters: {len(output_text):,}")
    print(f"Unique characters: {len(set(output_text)):,}")
    print(f"Non-alphanumeric symbols: {''.join(symbols)}")
    print(f"Saved: {output_path}")
    print("\nPDF note: selectable text is extracted; scanned/image-only PDFs require OCR.")


if __name__ == "__main__":
    main()
