"""
Free Unlimited Tools & Knowledge Engines (v2.2.0)
Includes zero-API-key open services:
1. ArXiv Academic Paper Search (physics, math, cs, stats, quant-bio, quant-fin)
2. Wikipedia Deep Knowledge & Encyclopedic Retrieval
3. Open-Meteo Real-Time Global Weather & Geocoding
"""
import json
import re
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


class ArxivSearchEngine:
    """Free, unlimited academic research paper search via official ArXiv API."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ArxivSearchClient/1.0"
    TIMEOUT = 8

    @classmethod
    def search(cls, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """Searches ArXiv repository and returns paper titles, authors, summaries, and links."""
        clean_q = query.strip()
        if not clean_q:
            return []

        results = []
        try:
            encoded = urllib.parse.quote(clean_q)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={max_results}&sortBy=relevance"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=cls.TIMEOUT) as resp:
                xml_data = resp.read().decode("utf-8", errors="replace")

            # Parse XML feed
            root = ET.fromstring(xml_data)
            # Atom namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)

            for entry in entries:
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                id_elem = entry.find("atom:id", ns)
                published_elem = entry.find("atom:published", ns)
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]

                title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"
                summary = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else "No abstract"
                link = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                published = published_elem.text[:10] if published_elem is not None and published_elem.text else ""

                results.append({
                    "title": title,
                    "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
                    "summary": summary[:380] + ("..." if len(summary) > 380 else ""),
                    "url": link,
                    "published": published,
                    "source": "ArXiv Open Repository",
                })
        except Exception:
            pass

        return results

    @classmethod
    def format_results(cls, query: str, results: List[Dict[str, str]]) -> str:
        """Renders clean Markdown output for ArXiv research queries."""
        if not results:
            return f"### ArXiv Research Papers: '{query}'\n\n*(No papers retrieved or service offline)*"

        lines = [f"### ArXiv Research Papers for: '{query}'\n"]
        for idx, r in enumerate(results, 1):
            lines.append(f"{idx}. **[{r['title']}]({r['url']})**")
            lines.append(f"   *Authors:* {r['authors']} ({r['published']})")
            lines.append(f"   {r['summary']}")
            lines.append(f"   *Source: {r['source']}* | [PDF Link]({r['url'].replace('abs', 'pdf')})\n")
        return "\n".join(lines)


class WikipediaEngine:
    """Free, unlimited Wikipedia REST API encyclopedic knowledge retriever."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WikipediaClient/1.0"
    TIMEOUT = 6

    @classmethod
    def summary(cls, topic: str) -> Optional[Dict[str, str]]:
        """Fetches full encyclopedic summary of a given concept or entity."""
        clean_topic = topic.strip()
        if not clean_topic:
            return None

        try:
            encoded = urllib.parse.quote(clean_topic.replace(" ", "_"))
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
            req = urllib.request.Request(url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=cls.TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))

            if "extract" in data and data["extract"]:
                page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                return {
                    "title": data.get("title", clean_topic),
                    "description": data.get("description", "Encyclopedic entry"),
                    "extract": data["extract"],
                    "url": page_url,
                    "source": "Wikipedia Open Encyclopedia",
                }
        except Exception:
            pass
        return None

    @classmethod
    def format_summary(cls, topic: str, result: Optional[Dict[str, str]]) -> str:
        if not result:
            return f"### Wikipedia Knowledge: '{topic}'\n\n*(Topic not found or connection unavailable)*"

        return (
            f"### Wikipedia: {result['title']}\n"
            f"*{result['description']}*\n\n"
            f"{result['extract']}\n\n"
            f"*Source: [{result['source']}]({result['url']})*"
        )


class WeatherEngine:
    """Free, unlimited global real-time weather and forecast via Open-Meteo API."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OpenMeteoClient/1.0"
    TIMEOUT = 6

    WEATHER_CODES = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
    }

    @classmethod
    def get_weather(cls, city: str) -> Optional[Dict[str, Any]]:
        """Geocodes city name and retrieves current weather conditions."""
        clean_city = city.strip()
        if not clean_city:
            return None

        try:
            # Step 1: Geocoding
            encoded = urllib.parse.quote(clean_city)
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=en&format=json"
            req = urllib.request.Request(geo_url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req, timeout=cls.TIMEOUT) as resp:
                geo_data = json.loads(resp.read().decode("utf-8", errors="replace"))

            if not geo_data.get("results"):
                return None

            location = geo_data["results"][0]
            lat = location["latitude"]
            lon = location["longitude"]
            city_name = location["name"]
            country = location.get("country", "")

            # Step 2: Current Weather
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            req2 = urllib.request.Request(weather_url, headers={"User-Agent": cls.USER_AGENT})
            with urllib.request.urlopen(req2, timeout=cls.TIMEOUT) as resp2:
                w_data = json.loads(resp2.read().decode("utf-8", errors="replace"))

            cw = w_data.get("current_weather", {})
            temp_c = cw.get("temperature")
            temp_f = round((temp_c * 9 / 5) + 32, 1) if temp_c is not None else None
            code = cw.get("weathercode", 0)
            desc = cls.WEATHER_CODES.get(code, "Clear")
            wind = cw.get("windspeed", 0.0)

            return {
                "city": city_name,
                "country": country,
                "latitude": lat,
                "longitude": lon,
                "temp_c": temp_c,
                "temp_f": temp_f,
                "condition": desc,
                "wind_speed_kmh": wind,
                "source": "Open-Meteo Free Global Weather",
            }
        except Exception:
            return None

    @classmethod
    def format_weather(cls, city: str, result: Optional[Dict[str, Any]]) -> str:
        if not result:
            return f"### Weather: '{city}'\n\n*(Could not find location or weather service is unavailable)*"

        return (
            f"### Current Weather in {result['city']}, {result['country']}\n\n"
            f"- **Temperature:** {result['temp_c']}°C ({result['temp_f']}°F)\n"
            f"- **Conditions:** {result['condition']}\n"
            f"- **Wind Speed:** {result['wind_speed_kmh']} km/h\n"
            f"- **Coordinates:** {result['latitude']}°N, {result['longitude']}°E\n\n"
            f"*(Live data powered by Open-Meteo)*"
        )
