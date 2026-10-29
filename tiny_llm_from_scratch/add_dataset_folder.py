from pathlib import Path
import shutil

source = Path(r"D:\book making\llm_knowledge_expansion\datasets")
original = Path("data_merged_updated.txt")
output = Path("data_merged_expanded.txt")

extensions = {".txt", ".md", ".csv", ".rst"}

if not source.is_dir():
    raise SystemExit(f"Dataset folder not found: {source}")

if not original.is_file():
    raise SystemExit(f"Original dataset not found: {original.resolve()}")

files = sorted(
    p for p in source.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

if not files:
    raise SystemExit("No supported text files found in the source folder.")

if output.exists():
    backup = output.with_suffix(".txt.backup")
    shutil.copy2(output, backup)
    print(f"Backed up previous expanded dataset to: {backup}")

count = 0
total_chars = 0

with output.open("w", encoding="utf-8", errors="replace") as out:
    with original.open("r", encoding="utf-8-sig", errors="replace") as src:
        shutil.copyfileobj(src, out, length=1024 * 1024)

    out.write("\n\n")

    for path in files:
        print(f"Adding: {path}")
        out.write(f"\n\n--- SOURCE: {path.name} ---\n\n")

        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as inp:
                while True:
                    chunk = inp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    total_chars += len(chunk)
            count += 1
        except OSError as exc:
            print(f"SKIPPED: {path} ({exc})")

print(f"\nFiles added: {count}")
print(f"Characters from added files: {total_chars:,}")
print(f"Expanded dataset: {output.resolve()}")
print(f"Expanded dataset size: {output.stat().st_size:,} bytes")
