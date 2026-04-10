"""
AgriSense-AI — agents/web_agent.py
Web search agent using DuckDuckGo Search.
"""

import sys
import urllib.parse
from ddgs import DDGS


def web_agent(query: str):
    """
    Fetches real-time web context without geographical bias.
    """
    # Simply optimize query for agriculture context without forcing India or strict bounds
    opt_query = f"{query} agriculture farming"
    
    try:
        ddgs = DDGS()
        # text() generates a generator, wrapping in list captures first few results
        results = list(ddgs.text(opt_query, max_results=5))

        if not results:
            return "No relevant web supplemental data found."

        formatted = []
        for r in results[:3]:
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
            formatted.append(f"Source: {domain} | {title}\nSummary: {clean_body}")

        return "\n\n".join(formatted)

    except Exception as e:
        print(f"--- LOG: Web search error: {str(e)} ---", file=sys.stderr)
        return "No relevant web supplemental data found."

