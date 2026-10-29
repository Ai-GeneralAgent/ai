"""
Retrieval-Augmented Generation (RAG) Module (Section 12)
Inverted index BM25 search, document extraction, and citation formatting.
"""
from ..rag.pipeline import LocalRAGPipeline, DocumentChunk

__all__ = ["LocalRAGPipeline", "DocumentChunk"]
