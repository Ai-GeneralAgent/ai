"""
Headless Web Browser Engine & Web Content Extractor
Provides zero-external-dependency web browsing, HTML-to-text parsing, link extraction,
and search-and-browse capabilities using Python's standard library.
"""
from html.parser import HTMLParser
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class CleanHTMLTextExtractor(HTMLParser):
    """Parses HTML into clean readable text, extracting title, headings, paragraphs, and links."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "header", "footer", "nav", "aside", "style"}

    def __init__(self, base_url: str = ""):
        super().__init__()
        self.base_url = base_url
        self.title: str = ""
        self.in_title = False
        self.skip_stack = 0
        self.text_chunks: List[str] = []
        self.links: List[Dict[str, str]] = []
        self.current_link_href: Optional[str] = None
        self.current_link_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag_lower = tag.lower()
        if tag_lower in self.SKIP_TAGS:
            self.skip_stack += 1
            return
        if self.skip_stack > 0:
            return

        if tag_lower == "title":
            self.in_title = True
        elif tag_lower in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "article", "section"):
            self.text_chunks.append("\n")
        elif tag_lower == "br":
            self.text_chunks.append("\n")
        elif tag_lower == "a":
            attr_dict = dict(attrs)
            href = attr_dict.get("href")
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                abs_url = urllib.parse.urljoin(self.base_url, href)
                self.current_link_href = abs_url
                self.current_link_text = []

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in self.SKIP_TAGS:
            if self.skip_stack > 0:
                self.skip_stack -= 1
            return
        if self.skip_stack > 0:
            return

        if tag_lower == "title":
            self.in_title = False
        elif tag_lower in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self.text_chunks.append("\n")
        elif tag_lower == "a" and self.current_link_href:
            anchor_text = " ".join("".join(self.current_link_text).split())
            if anchor_text:
                self.links.append({"text": anchor_text, "url": self.current_link_href})
            self.current_link_href = None
            self.current_link_text = []

    def handle_data(self, data: str):
        if self.skip_stack > 0:
            return
        if self.in_title:
            self.title += data
            return

        if self.current_link_href:
            self.current_link_text.append(data)

        self.text_chunks.append(data)

    def get_clean_text(self) -> str:
        raw_text = "".join(self.text_chunks)
        # Collapse repeated spaces and clean up line breaks
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
        # Remove consecutive empty lines
        clean_lines: List[str] = []
        for line in lines:
            if line:
                clean_lines.append(line)
            elif clean_lines and clean_lines[-1] != "":
                clean_lines.append("")
        return "\n".join(clean_lines).strip()


class WebBrowserEngine:
    """Headless web browser for live document retrieval and web page interaction."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 AntigravityResearch/2.2"
    DEFAULT_TIMEOUT_SEC = 8.0

    @classmethod
    def browse_url(cls, url: str, max_chars: int = 5000) -> Dict[str, Any]:
        """
        Fetches an HTTP/HTTPS web page, parses HTML DOM, strips scripts/styles,
        and returns clean structured text, title, and links.
        """
        target_url = url.strip()
        if not target_url.startswith(("http://", "https://")):
            target_url = "https://" + target_url

        try:
            req = urllib.request.Request(
                target_url,
                headers={
                    "User-Agent": cls.USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            )
            with urllib.request.urlopen(req, timeout=cls.DEFAULT_TIMEOUT_SEC) as resp:
                status_code = resp.getcode()
                content_type = resp.headers.get("Content-Type", "")
                charset = resp.headers.get_content_charset() or "utf-8"
                raw_bytes = resp.read()

            try:
                html_text = raw_bytes.decode(charset, errors="replace")
            except Exception:
                html_text = raw_bytes.decode("utf-8", errors="replace")

            extractor = CleanHTMLTextExtractor(base_url=target_url)
            extractor.feed(html_text)
            clean_text = extractor.get_clean_text()
            title = extractor.title.strip() or target_url

            truncated = False
            if len(clean_text) > max_chars:
                clean_text = clean_text[:max_chars] + f"\n\n[... Truncated: page has {len(clean_text):,} total chars ...]"
                truncated = True

            return {
                "success": True,
                "url": target_url,
                "title": title,
                "status_code": status_code,
                "content": clean_text,
                "links": extractor.links[:30],
                "content_length": len(clean_text),
                "truncated": truncated,
            }

        except urllib.error.HTTPError as e:
            return {
                "success": False,
                "url": target_url,
                "status_code": e.code,
                "error": f"HTTP Error {e.code}: {e.reason}",
            }
        except urllib.error.URLError as e:
            return {
                "success": False,
                "url": target_url,
                "error": f"Connection Failed: {e.reason}",
            }
        except Exception as e:
            return {
                "success": False,
                "url": target_url,
                "error": f"Browser Error: {str(e)}",
            }

    @classmethod
    def search_and_browse(cls, query: str, max_chars: int = 4000) -> Dict[str, Any]:
        """
        Searches web using DuckDuckGo, then immediately browses and extracts the top result.
        Returns combined search metadata and full page content.
        """
        from .web_search import DuckDuckGoSearchEngine
        search_results = DuckDuckGoSearchEngine.search(query, max_results=3)
        if not search_results:
            return {
                "success": False,
                "query": query,
                "error": "No search results found to browse.",
            }

        top_result = search_results[0]
        page = cls.browse_url(top_result["url"], max_chars=max_chars)
        page["search_query"] = query
        page["search_snippet"] = top_result.get("snippet", "")
        page["search_results"] = search_results
        return page

    @classmethod
    def format_browser_display(cls, page_dict: Dict[str, Any]) -> str:
        """Formats browser result into a clean terminal/markdown card."""
        if not page_dict.get("success"):
            return f"[BROWSER ERROR] URL: {page_dict.get('url')} | Error: {page_dict.get('error')}"

        lines = [
            f"=== [BROWSER: {page_dict.get('title')}] ===",
            f"URL:    {page_dict.get('url')} (Status: {page_dict.get('status_code')})",
            f"Length: {page_dict.get('content_length'):,} characters",
            "-" * 60,
            page_dict.get("content", ""),
            "-" * 60,
        ]
        links = page_dict.get("links", [])
        if links:
            lines.append("Key Page Links:")
            for l in links[:5]:
                lines.append(f"  - [{l['text']}] -> {l['url']}")
        return "\n".join(lines)
