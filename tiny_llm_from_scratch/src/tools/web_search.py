"""
DuckDuckGo Web Search & Live Knowledge Engine
Provides zero-dependency real-time web retrieval via DuckDuckGo Instant Answer API
and Encyclopedic Knowledge Graphs with live citations and snippet extraction.
"""
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request


class DuckDuckGoSearchEngine:
    """Robust DuckDuckGo live web search engine with zero external API key requirements."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    TIMEOUT_SECONDS = 6

    @classmethod
    def search(cls, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a real-time web query via DuckDuckGo API and Wikipedia knowledge graphs.
        Returns a list of structured search results: [{"title": ..., "snippet": ..., "url": ..., "source": ...}]
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        results: List[Dict[str, str]] = []
        seen_urls = set()

        # Step 1: DuckDuckGo Instant Answer API
        try:
            encoded = urllib.parse.quote(clean_query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=cls.TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))

            # Primary Abstract
            heading = data.get("Heading") or clean_query
            abstract = data.get("Abstract")
            abstract_url = data.get("AbstractURL")
            source = data.get("AbstractSource", "DuckDuckGo")

            if abstract and abstract_url:
                results.append({
                    "title": heading,
                    "snippet": abstract,
                    "url": abstract_url,
                    "source": f"DuckDuckGo ({source})",
                })
                seen_urls.add(abstract_url)

            # Related Topics
            for item in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                text = item.get("Text", "")
                first_url = item.get("FirstURL", "")
                if text and first_url and first_url not in seen_urls:
                    # Split first phrase as title if possible
                    title_part = text.split(" - ")[0] if " - " in text else text[:40]
                    results.append({
                        "title": title_part,
                        "snippet": text,
                        "url": first_url,
                        "source": "DuckDuckGo Knowledge",
                    })
                    seen_urls.add(first_url)

        except Exception:
            pass  # Fall through to companion provider

        # Step 2: Encyclopedic Knowledge Companion (if DuckDuckGo had fewer results)
        if len(results) < max_results:
            try:
                # Query Wikipedia summary API
                topic = clean_query.split()[0] if len(clean_query.split()) > 1 else clean_query
                wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
                req = urllib.request.Request(wiki_url, headers={"User-Agent": cls.USER_AGENT})
                with urllib.request.urlopen(req, timeout=cls.TIMEOUT_SECONDS) as resp:
                    wdata = json.loads(resp.read().decode("utf-8", errors="replace"))
                
                wtitle = wdata.get("title")
                wextract = wdata.get("extract")
                wpage = wdata.get("content_urls", {}).get("desktop", {}).get("page")
                if wextract and wpage and wpage not in seen_urls:
                    results.append({
                        "title": wtitle or topic,
                        "snippet": wextract[:350],
                        "url": wpage,
                        "source": "Wikipedia Knowledge Base",
                    })
                    seen_urls.add(wpage)
            except Exception:
                pass

        return results[:max_results]

    @classmethod
    def format_search_results(cls, query: str, results: List[Dict[str, str]]) -> str:
        """Formats structured web search results into clean, professional Markdown."""
        if not results:
            return (
                f"### DuckDuckGo Web Search: '{query}'\n\n"
                f"*(No live online results found or network connection unavailable. Grounding with local knowledge base.)*"
            )

        output = [f"### DuckDuckGo Web Search Results: '{query}'\n"]
        for idx, item in enumerate(results, 1):
            title = item.get("title", f"Result {idx}")
            snippet = item.get("snippet", "").strip()
            url = item.get("url", "")
            source = item.get("source", "DuckDuckGo")

            output.append(f"{idx}. **[{title}]({url})**")
            output.append(f"   {snippet}")
            output.append(f"   *Source: {source}* | [Link]({url})\n")

        return "\n".join(output)
