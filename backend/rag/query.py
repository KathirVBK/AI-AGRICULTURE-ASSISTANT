"""
AgriSense-AI — rag/query.py
Pure RAG retrieval + LLM synthesis.
The pipeline (core/pipeline.py) is the rules engine.
This module is responsible ONLY for knowledge retrieval and response generation.
"""

from rag.rag_pipeline import retrieve
from rag.llm import generate_response
from utils.validator import validate_input, classify_query, enforce_grounding
from agents.web_agent import web_agent


def retrieve_context(
    query: str,
    ml_report: str = "",
    history: list = None,
    is_subcall: bool = False
) -> str:
    """
    Fetches RAG + web context and generates an LLM response.
    Returns a raw string (no UI formatting).
    This is called by rag_agent and by pipeline.py internally.
    """
    # Validate input
    err = validate_input(query)
    if err:
        return err

    intent = classify_query(query)

    # Retrieve from local vector store
    try:
        rag_text, _ = retrieve(f"{query} agriculture", k=5)
    except Exception:
        rag_text = ""

    # Retrieve from web
    web_text = web_agent(query)

    # Build clean context block
    context_parts = []
    if ml_report and ml_report.strip():
        context_parts.append(f"[Pre-validated Crops]\n{ml_report.strip()}")
    if rag_text and rag_text.strip():
        context_parts.append(f"[Knowledge Base]\n{rag_text.strip()}")
    if web_text and web_text.strip() and "No relevant" not in web_text:
        context_parts.append(f"[Web Search]\n{web_text.strip()}")

    context = "\n\n".join(context_parts) if context_parts else "No supplemental context found."

    # Generate response
    raw = generate_response(
        query=query,
        context=context,
        intent=intent,
        history=history
    )

    # Grounding check — if response doesn't match context, retry once in strict mode
    if not enforce_grounding(raw, context):
        raw = generate_response(
            query=query,
            context=context,
            intent=intent,
            history=history,
            strict=True
        )

    return raw.strip()
