from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_name: str
    source_path: str
    text: str
    chunk_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    citation: str


class LocalRAGPipeline:
    """
    Lightweight, deterministic local RAG pipeline:
    Chunking -> Inverted Index -> TF-IDF / BM25 Scoring -> Citation Formatting.
    Runs locally on CPU with zero external API dependencies.
    """

    def __init__(self, chunk_size_words: int = 200, overlap_words: int = 40):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.chunks: List[DocumentChunk] = []
        self.doc_freqs: Dict[str, int] = {}
        self.chunk_term_counts: List[Dict[str, int]] = []
        self.chunk_lengths: List[int] = []
        self.avg_chunk_len: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def add_document(self, file_path: Union[str, Path], text: Optional[str] = None):
        """Chunks and indexes a document."""
        p = Path(file_path)
        if text is None:
            if not p.is_file():
                return
            text = p.read_text(encoding="utf-8-sig", errors="replace")

        words = text.split()
        if not words:
            return

        step = max(1, self.chunk_size_words - self.overlap_words)
        chunk_idx = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size_words]
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{p.stem}_c{chunk_idx}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_name=p.name,
                source_path=str(p),
                text=chunk_text,
                chunk_idx=chunk_idx,
            )
            self.chunks.append(chunk)

            # Inverted index stats
            terms = self._tokenize(chunk_text)
            counts: Dict[str, int] = {}
            for t in terms:
                counts[t] = counts.get(t, 0) + 1

            for unique_t in counts:
                self.doc_freqs[unique_t] = self.doc_freqs.get(unique_t, 0) + 1

            self.chunk_term_counts.append(counts)
            self.chunk_lengths.append(len(terms))
            chunk_idx += 1

        if self.chunk_lengths:
            self.avg_chunk_len = sum(self.chunk_lengths) / len(self.chunk_lengths)

    def search(
        self, query: str, top_k: int = 3, min_score: float = 0.05
    ) -> List[RetrievalResult]:
        """BM25 search across indexed document chunks."""
        query_terms = self._tokenize(query)
        if not query_terms or not self.chunks:
            return []

        N = len(self.chunks)
        k1 = 1.5
        b = 0.75
        scores = [0.0] * N

        for term in query_terms:
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            # Standard Lucene/BM25 IDF
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

            for idx, counts in enumerate(self.chunk_term_counts):
                tf = counts.get(term, 0)
                if tf > 0:
                    L = self.chunk_lengths[idx] / max(1.0, self.avg_chunk_len)
                    score_term = idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * L))
                    scores[idx] += score_term

        # Rank and filter
        scored_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []

        for idx, score in scored_pairs[:top_k]:
            if score < min_score:
                break
            chunk = self.chunks[idx]
            citation = f"[{chunk.doc_name}#chunk_{chunk.chunk_idx}]"
            results.append(RetrievalResult(chunk=chunk, score=round(score, 3), citation=citation))

        return results

    def format_context_for_prompt(self, query: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """Retrieves chunks and formats them with citations as evidence for generation."""
        results = self.search(query, top_k=top_k)
        if not results:
            return "No relevant documents found.", []

        evidence_lines = ["### Retrieved Evidence:"]
        citations = []
        for r in results:
            evidence_lines.append(f"{r.citation}: {r.chunk.text.strip()}")
            citations.append(r.citation)

        return "\n\n".join(evidence_lines), citations
