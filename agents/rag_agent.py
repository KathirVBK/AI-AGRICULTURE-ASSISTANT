"""
AgriSense-AI — rag_agent.py
"""

# agents/rag_agent.py
# Thin wrapper: delegates to rag/query.py's retrieve_context.
from rag.query import retrieve_context


def rag_agent(query: str, history: list = None, is_subcall: bool = False) -> str:
    """
    Returns a raw text answer from the RAG + LLM pipeline.
    """
    try:
        return retrieve_context(query, history=history, is_subcall=is_subcall)
    except Exception as e:
        return f"⚠️ Knowledge retrieval error: {str(e)}"
