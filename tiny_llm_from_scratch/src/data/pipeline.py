import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .manifest import DatasetManifest, DocumentRecord


class CorpusPipeline:
    """
    Reproducible Corpus Preparation Pipeline:
    Extraction -> Normalization -> Quality Checks -> Deduplication -> Splits -> Manifest.
    """

    DOMAIN_KEYWORDS = {
        "programming": ["python", "def ", "class ", "function", "import ", "const ", "return", "int ", "void ", "algorithm"],
        "mathematics": ["theorem", "proof", "lemma", "equation", "matrix", "integral", "derivative", "probability", "algebra"],
        "science": ["physics", "biology", "chemistry", "quantum", "molecule", "organism", "experiment", "gravity", "cell"],
        "ai": ["transformer", "neural", "gradient", "loss", "backpropagation", "token", "attention", "dataset", "deep learning"],
        "business": ["market", "revenue", "investing", "profit", "capital", "shares", "finance", "business", "valuation"],
        "literature": ["chapter", "replied", "exclaimed", "cried", "whispered", "countenance", "thee", "thou"],
        "conversation": ["user:", "assistant:", "question:", "answer:", "q:", "a:", "dialogue", "chat"],
    }

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path
        self.manifest = DatasetManifest.load(manifest_path) if (manifest_path and manifest_path.is_file()) else DatasetManifest()
        self.seen_hashes: Set[str] = set()

    def extract_text(self, file_path: Path) -> str:
        """Extracts text from TXT, MD, JSONL, or PDF files safely."""
        suffix = file_path.suffix.lower()
        if suffix in [".txt", ".md", ".py"]:
            return file_path.read_text(encoding="utf-8-sig", errors="replace")
        elif suffix == ".jsonl":
            lines = []
            with file_path.open("r", encoding="utf-8-sig", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, str):
                            lines.append(obj)
                        elif isinstance(obj, dict):
                            txt = obj.get("text") or obj.get("content") or obj.get("prompt", "") + " " + obj.get("response", "")
                            if txt:
                                lines.append(txt)
                    except Exception:
                        continue
            return "\n".join(lines)
        elif suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages)
            except Exception as e:
                print(f"Warning: PDF extraction failed for {file_path}: {e}")
                return ""
        return ""

    def normalize_text(self, text: str) -> str:
        """Normalizes line endings and whitespace while preserving code, math, and unicode."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive blank lines (>3) down to 2
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text.strip()

    def check_quality(self, text: str) -> Tuple[bool, Optional[str]]:
        """Checks text quality. Flags empty, repetitive, or corrupt documents."""
        if len(text.strip()) < 50:
            return False, "Too short (<50 characters)"

        # Check for excessive repetition (e.g. same 20-char substring repeated 50 times)
        sample = text[:1000]
        if len(sample) > 100:
            words = sample.split()
            if len(words) > 10 and len(set(words)) / len(words) < 0.15:
                return False, "Excessive word repetition"

        return True, None

    def classify_domain(self, text: str) -> str:
        """Classifies document into an educational/technical domain."""
        lower = text.lower()[:5000]
        domain_scores = {}
        for domain, kw_list in self.DOMAIN_KEYWORDS.items():
            score = sum(lower.count(kw) for kw in kw_list)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"

    def process_document(
        self,
        file_path: Path,
        split: str = "train",
        domain_override: Optional[str] = None,
    ) -> Optional[Tuple[DocumentRecord, str]]:
        """Processes a single document and updates the manifest."""
        raw_text = self.extract_text(file_path)
        norm_text = self.normalize_text(raw_text)

        is_valid, reason = self.check_quality(norm_text)
        fingerprint = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()[:16]

        doc_id = f"doc_{file_path.stem}_{fingerprint[:8]}"
        domain = domain_override or self.classify_domain(norm_text)

        if not is_valid:
            record = DocumentRecord(
                doc_id=doc_id,
                filename=str(file_path),
                domain=domain,
                char_count=len(norm_text),
                split="excluded",
                exclusion_reason=reason,
                fingerprint=fingerprint,
            )
            self.manifest.add_document(record)
            return None

        # Exact duplicate detection
        if fingerprint in self.seen_hashes:
            record = DocumentRecord(
                doc_id=doc_id,
                filename=str(file_path),
                domain=domain,
                char_count=len(norm_text),
                duplicate_status="exact_dup",
                split="excluded",
                exclusion_reason="Exact duplicate of existing document",
                fingerprint=fingerprint,
            )
            self.manifest.add_document(record)
            return None

        self.seen_hashes.add(fingerprint)
        record = DocumentRecord(
            doc_id=doc_id,
            filename=str(file_path),
            domain=domain,
            char_count=len(norm_text),
            split=split,
            fingerprint=fingerprint,
        )
        self.manifest.add_document(record)
        return record, norm_text

