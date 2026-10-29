from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    domain: str
    char_count: int
    token_count: int = 0
    language: str = "en"
    license_info: str = "unknown"
    duplicate_status: str = "unique"  # "unique", "exact_dup", "near_dup"
    split: str = "train"              # "train", "val", "test", "excluded"
    exclusion_reason: Optional[str] = None
    fingerprint: str = ""


class DatasetManifest:
    """Tracks document provenance, quality status, domain labels, and train/val/test splits."""
    def __init__(self):
        self.records: Dict[str, DocumentRecord] = {}

    def add_document(self, record: DocumentRecord):
        self.records[record.doc_id] = record

    def get_summary(self) -> Dict[str, Any]:
        domains: Dict[str, int] = {}
        splits: Dict[str, int] = {}
        total_chars = 0
        total_tokens = 0

        for r in self.records.values():
            if r.split != "excluded":
                domains[r.domain] = domains.get(r.domain, 0) + 1
                splits[r.split] = splits.get(r.split, 0) + 1
                total_chars += r.char_count
                total_tokens += r.token_count

        return {
            "total_documents": len(self.records),
            "active_documents": sum(1 for r in self.records.values() if r.split != "excluded"),
            "domains": domains,
            "splits": splits,
            "total_chars": total_chars,
            "total_tokens": total_tokens,
        }

    def save(self, path: Path):
        data = {doc_id: asdict(rec) for doc_id, rec in self.records.items()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        manifest = cls()
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for doc_id, rec_data in data.items():
                manifest.records[doc_id] = DocumentRecord(**rec_data)
        return manifest

