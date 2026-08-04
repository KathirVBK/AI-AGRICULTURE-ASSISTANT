"""
AgriSense-AI — agents/web_agent.py
Web search agent using DuckDuckGo Search.
"""

import sys
import urllib.parse
import re
from ddgs import DDGS


def web_agent(query: str):
    """
    Fetches real-time web context without geographical bias.
    """
    # Clean and truncate long user queries (e.g. soil test reports) for web search
    clean_q = re.sub(r'[\r\n]+', ' ', query).strip()[:150]
    opt_query = f"{clean_q} agriculture farming"
    
    try:
        # Use a context manager to ensure DDGS resources are cleaned up
        with DDGS(timeout=7) as ddgs:
            # text() generates a generator, wrapping in list captures first few results
            results = list(ddgs.text(opt_query, max_results=3))

        if not results:
            return "No relevant web supplemental data found."

        formatted = []
        for i, r in enumerate(results[:3], 1):
            title = r.get("title", "Untitled")
            body = r.get("body", r.get("description", ""))
            link = r.get("href", "")

            # safely get domain
            try:
                domain = urllib.parse.urlparse(link).netloc
            except Exception:
                domain = "web"

            # Clean up the body text to avoid huge blob injections
            clean_body = body.strip().replace("\n", " ")
            formatted.append(f"Web Source [{i}]: [{title}]({link}) | Domain: {domain}\nSummary: {clean_body}")

        return "\n\n".join(formatted)

    except Exception as e:
        print(f"--- LOG: Web search error: {str(e)} ---", file=sys.stderr)
        return "No relevant web supplemental data found."

