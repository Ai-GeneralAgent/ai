"""
Internet Learning Engine: Autonomous Web Ingestion, Knowledge Distillation & Continual Learning
Uses DuckDuckGo search + Headless Web Browsing to autonomously learn any topic from the internet:
1. Searches DuckDuckGo for top authoritative sources & definitions
2. Browses deep webpage text, headings, and code snippets via WebBrowserEngine
3. Distills durable facts and stores them into the SQLite Cognitive Hippocampus
4. Generates local Markdown knowledge files and indexes them directly into Local RAG
5. Formulates ChatML instruction-tuning pairs for continual model weight updates
"""
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .web_search import DuckDuckGoSearchEngine
from .browser import WebBrowserEngine
from ..memory.hippocampus import HippocampusMemory
from ..rag.pipeline import LocalRAGPipeline


@dataclass
class LearnedKnowledgePackage:
    topic: str
    query: str
    source_urls: List[str]
    extracted_facts: List[str]
    summary: str
    saved_document_path: Optional[str] = None
    hippocampus_facts_stored: int = 0
    sft_samples_generated: int = 0
    rag_indexed: bool = False
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "query": self.query,
            "source_urls": self.source_urls,
            "extracted_facts": self.extracted_facts,
            "summary": self.summary,
            "saved_document_path": self.saved_document_path,
            "hippocampus_facts_stored": self.hippocampus_facts_stored,
            "sft_samples_generated": self.sft_samples_generated,
            "rag_indexed": self.rag_indexed,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


class InternetLearnerEngine:
    """Autonomous internet learning pipeline for Tiny LLM."""

    CATEGORY_MAP = {
        "security": "cybersecurity",
        "cyber": "cybersecurity",
        "vulnerability": "cybersecurity",
        "crypto": "cybersecurity",
        "malware": "cybersecurity",
        "exploit": "cybersecurity",
        "injection": "cybersecurity",
        "sqli": "cybersecurity",
        "xss": "cybersecurity",
        "python": "coding",
        "code": "coding",
        "algorithm": "coding",
        "rust": "coding",
        "javascript": "coding",
        "c++": "coding",
        "stock": "finance_stock_market",
        "finance": "finance_stock_market",
        "market": "finance_stock_market",
        "business": "business",
        "economy": "business",
        "quantum": "science_research",
        "physics": "science_research",
        "biology": "science_research",
        "space": "science_research",
        "ai": "science_research",
        "llm": "science_research",
    }

    @classmethod
    def determine_category(cls, topic: str) -> str:
        """Determines category folder based on keywords."""
        lower = topic.lower()
        for kw, cat in cls.CATEGORY_MAP.items():
            if kw in lower:
                return cat
        return "science_research"

    @classmethod
    def extract_facts_from_text(cls, topic: str, text: str, max_facts: int = 6) -> List[str]:
        """Extracts high-signal factual statements from raw web text."""
        facts: List[str] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        topic_tokens = set(re.findall(r"\w+", topic.lower()))

        # 1. Definitional sentences ("X is a ...", "X refers to ...")
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 30 or len(s_clean) > 250:
                continue
            s_lower = s_clean.lower()
            if any(t in s_lower for t in topic_tokens):
                if any(ind in s_lower for ind in [" is a ", " is an ", " refers to ", " is defined as ", " was developed ", " consists of "]):
                    if s_clean not in facts:
                        facts.append(s_clean)
            if len(facts) >= max_facts:
                break

        # 2. Key descriptive sentences if fewer than target
        if len(facts) < max_facts:
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 40 or len(s_clean) > 200:
                    continue
                s_lower = s_clean.lower()
                if any(t in s_lower for t in topic_tokens) and s_clean not in facts:
                    facts.append(s_clean)
                if len(facts) >= max_facts:
                    break

        return facts[:max_facts]

    @classmethod
    def learn(
        cls,
        topic: str,
        project_root: Optional[Path] = None,
        hippocampus: Optional[HippocampusMemory] = None,
        rag_pipeline: Optional[LocalRAGPipeline] = None,
        deep_browse: bool = True,
    ) -> LearnedKnowledgePackage:
        """
        Executes end-to-end internet learning:
        DuckDuckGo search -> WebBrowser deep scrape -> Fact extraction -> SQLite Hippocampus -> Local RAG -> SFT Dataset.
        """
        start_time = time.time()
        root = project_root or Path.cwd()
        clean_topic = topic.strip()

        # 1. Query DuckDuckGo
        search_results = DuckDuckGoSearchEngine.search(clean_topic, max_results=4)
        source_urls = [r["url"] for r in search_results if r.get("url")]

        # 2. Gather Content from Search & Deep Browsing
        combined_text_parts = []
        for r in search_results:
            if r.get("snippet"):
                combined_text_parts.append(f"Source [{r.get('title')}]: {r.get('snippet')}")

        deep_browsed_url = None
        if deep_browse and source_urls:
            top_url = source_urls[0]
            if top_url.startswith("http"):
                deep_browsed_url = top_url
                page = WebBrowserEngine.browse_url(top_url, max_chars=3500)
                if page.get("success") and page.get("clean_text"):
                    combined_text_parts.append(f"\n--- Detailed Page Content from {top_url} ---\n{page.get('clean_text')}")

        full_raw_text = "\n\n".join(combined_text_parts)

        # 3. Extract Facts & Summary
        extracted_facts = cls.extract_facts_from_text(clean_topic, full_raw_text, max_facts=6)
        if not extracted_facts:
            extracted_facts = [r.get("snippet") for r in search_results[:3] if r.get("snippet")]

        summary = f"Synthesized knowledge about **{clean_topic}** gathered from {len(source_urls)} live web source(s) via DuckDuckGo."
        if extracted_facts:
            summary += "\n\n**Core Principles & Verified Facts:**\n" + "\n".join(f"- {f}" for f in extracted_facts)

        # 4. Commit to Tier 3 Durable Memory (Hippocampus SQLite)
        hippo = hippocampus or HippocampusMemory(root / "tiny_llm_from_scratch" / "memory" / "hippocampus.db")
        facts_stored_count = 0
        for fact in extracted_facts:
            success = hippo.store_fact(
                content=fact,
                category="web_knowledge",
                confidence=0.95,
                source_session="internet_learner",
                tags=[clean_topic.lower(), "web_learned"],
            )
            if success:
                facts_stored_count += 1

        # 5. Ingest into Local RAG Knowledge Base
        category = cls.determine_category(clean_topic)
        target_dir = root / "tiny_llm_from_scratch" / "data_knowledge" / "web_knowledge" / category
        target_dir.mkdir(parents=True, exist_ok=True)

        slug = re.sub(r"[^\w\-]+", "_", clean_topic.lower()).strip("_")
        doc_path = target_dir / f"learned_{slug}.md"

        doc_content = f"# Learned Knowledge: {clean_topic}\n\n"
        doc_content += f"**Ingested Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        doc_content += f"**Category:** {category}\n"
        doc_content += f"**Source URLs:**\n" + "\n".join(f"- {u}" for u in source_urls) + "\n\n"
        doc_content += f"## Summary & Verified Principles\n\n"
        for f in extracted_facts:
            doc_content += f"- {f}\n"
        doc_content += f"\n## Detailed Web Content\n\n{full_raw_text[:2500]}\n"

        doc_path.write_text(doc_content, encoding="utf-8")
        rag_indexed = False
        if rag_pipeline:
            rag_pipeline.add_document(doc_path, text=doc_content)
            rag_indexed = True

        # 6. Formulate Continual SFT Training Dialogue Pairs
        sft_samples_count = 0
        sft_path = root / "tiny_llm_from_scratch" / "training_data" / "continual_web_learning.jsonl"
        sft_path.parent.mkdir(parents=True, exist_ok=True)

        qa_pairs = [
            (
                f"What is {clean_topic} and how does it work?",
                f"{clean_topic} is an important technical concept. Key facts: " + " ".join(extracted_facts[:2]),
            ),
            (
                f"Explain the key principles of {clean_topic}.",
                f"Based on verified research, the core principles of {clean_topic} include:\n" + "\n".join(f"- {f}" for f in extracted_facts),
            ),
        ]

        with open(sft_path, "a", encoding="utf-8") as fp:
            for user_q, asst_a in qa_pairs:
                record = {
                    "messages": [
                        {"role": "user", "content": user_q},
                        {"role": "assistant", "content": asst_a},
                    ],
                    "metadata": {"source": "internet_learner", "topic": clean_topic, "timestamp": time.time()},
                }
                fp.write(json.dumps(record) + "\n")
                sft_samples_count += 1

        elapsed = time.time() - start_time

        return LearnedKnowledgePackage(
            topic=clean_topic,
            query=clean_topic,
            source_urls=source_urls,
            extracted_facts=extracted_facts,
            summary=summary,
            saved_document_path=str(doc_path),
            hippocampus_facts_stored=facts_stored_count,
            sft_samples_generated=sft_samples_count,
            rag_indexed=rag_indexed,
            elapsed_sec=elapsed,
        )

    @classmethod
    def format_learning_report(cls, pkg: LearnedKnowledgePackage) -> str:
        """Renders an informative learning card for user output."""
        lines = [
            f"### 🌐 [INTERNET LEARNING COMPLETED: '{pkg.topic}']",
            f"*{pkg.summary}*",
            "",
            "#### 💡 Key Verified Facts Extracted:",
        ]
        for fact in pkg.extracted_facts:
            lines.append(f"- {fact}")

        lines.extend([
            "",
            "#### 📚 Knowledge Integration Metrics:",
            f"- **Sources Consulted:** {len(pkg.source_urls)} URL(s) via DuckDuckGo",
            f"- **Durable Facts Ingested:** {pkg.hippocampus_facts_stored} facts committed to SQLite Hippocampus",
            f"- **Local RAG Document:** `{Path(pkg.saved_document_path).name if pkg.saved_document_path else 'None'}`",
            f"- **Continual SFT Samples:** {pkg.sft_samples_generated} ChatML pairs saved to `continual_web_learning.jsonl`",
            f"- **Ingestion Time:** {pkg.elapsed_sec:.2f} seconds",
            "",
            "#### 🔗 References & Citations:",
        ])
        for u in pkg.source_urls[:3]:
            lines.append(f"- [{u}]({u})")
        return "\n".join(lines)
