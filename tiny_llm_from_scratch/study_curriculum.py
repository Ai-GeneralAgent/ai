#!/usr/bin/env python3
"""
Curriculum Study Pipeline for Grades 1 to 12.
Studies textbooks like a human student:
1. Ingests books from data_curriculum/grade_1 to grade_12.
2. Extracts core concepts, formulas, and definitions.
3. Generates question-and-answer flashcards and study notes.
4. Updates RAG knowledge base and builds progressive curriculum training corpus.
"""
import argparse
import json
import os
from pathlib import Path
import sys

# OpenBLAS / MKL thread protection on Windows CPU
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.curriculum import CurriculumStudyEngine
from src.rag.pipeline import LocalRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Study textbooks from Grades 1 to 12 like a human student.")
    parser.add_argument("--dir", default="data_curriculum", help="Directory containing grade folders or books")
    parser.add_argument("--book", default=None, help="Specific book file to study (PDF, TXT, MD)")
    parser.add_argument("--grade", default=None, help="Grade level hint (e.g. grade_1 ... grade_12)")
    parser.add_argument("--subject", default="general", help="Subject (math, science, history, english, etc.)")
    args = parser.parse_args()

    curriculum_dir = PROJECT_ROOT / args.dir
    curriculum_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RAG
    rag = LocalRAGPipeline()
    engine = CurriculumStudyEngine(curriculum_dir, rag_pipeline=rag)

    print("=" * 65)
    print("   CURRICULUM STUDY ENGINE (GRADES 1 - 12)")
    print("=" * 65)

    # Single book study
    if args.book:
        b_path = Path(args.book)
        if not b_path.is_absolute():
            b_path = PROJECT_ROOT / b_path
        if not b_path.is_file():
            print(f"Error: Book file not found: {b_path}")
            return
        print(f"\nStudying single textbook: {b_path.name}...")
        res = engine.study_book(b_path, grade_hint=args.grade, subject=args.subject)
        print(f"  • Grade: {res['grade_name']} (Difficulty: {res['difficulty']}/12)")
        print(f"  • Key Concepts: {res['concepts_learned']}")
        print(f"  • Flashcards: {res['flashcards_generated']}")
        print(f"  • Study Notes: {res['study_notes_path']}")
        return

    # Scan all grade folders
    grade_keys = [f"grade_{i}" for i in range(1, 13)]
    for gk in grade_keys:
        (curriculum_dir / gk).mkdir(exist_ok=True)

    supported_exts = [".pdf", ".txt", ".md"]
    found_books = []
    for ext in supported_exts:
        found_books.extend(list(curriculum_dir.rglob(f"*{ext}")))

    # Filter out generated notes and flashcards
    real_books = [b for b in found_books if "_study_notes" not in b.name and "flashcards" not in b.name]

    if not real_books:
        print(f"\nNo raw textbooks found in '{curriculum_dir.name}'.")
        print("To study textbooks:")
        print(f"  1. Place your books (PDF, TXT, MD) into:")
        print(f"     {curriculum_dir.resolve()}\\grade_1 through grade_12")
        print(f"  2. Or run: python study_curriculum.py --book <path_to_book.pdf> --grade grade_10")
        return

    print(f"\nFound {len(real_books)} textbook(s) to study across Grades 1-12:")
    total_concepts = 0
    total_flashcards = 0

    for book in sorted(real_books):
        grade_hint = None
        for gk in grade_keys:
            if gk in book.parts:
                grade_hint = gk
                break
        print(f"\nStudying: {book.name}...")
        try:
            res = engine.study_book(book, grade_hint=grade_hint, subject=args.subject)
            print(f"  • Classified: {res['grade_name']} (Level {res['difficulty']})")
            print(f"  • Concepts Extracted: {res['concepts_learned']}")
            print(f"  • Flashcards Generated: {res['flashcards_generated']}")
            total_concepts += res["concepts_learned"]
            total_flashcards += res["flashcards_generated"]
        except Exception as e:
            print(f"  • Error: {e}")

    print("\n" + "=" * 65)
    print(f"STUDY SESSION COMPLETE: {len(real_books)} books, {total_concepts} concepts, {total_flashcards} flashcards.")
    print("=" * 65)


if __name__ == "__main__":
    main()

