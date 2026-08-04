"""
AgriSense-AI — core/pipeline.py
The single authoritative entry point for all queries.

Retrieval hierarchy (sequential, not parallel):
  Tier 1 — Uploaded document (session soil report)         [highest trust]
  Tier 2 — Vector DB / internal knowledge base             [high trust]
  Tier 3 — Web search (only if Tiers 1+2 are insufficient) [medium trust]

Flow:
  Query → Validate → Guardrail → Impossible-Check → Extract Constraints
       → Filter Crops → Hierarchical Knowledge Fetch → LLM → Format Output
"""

import re
import logging
import concurrent.futures
from core.crop_engine import filter_crops, prioritize
from utils.validator import validate_input, extract_entities, hard_filter_crops, prioritize_crops
from utils.guardrails import guardrail_response
from rag.llm import generate_response, refine_response
from rag.rag_pipeline import retrieve
from agents.web_agent import web_agent
from utils.context_classifier import build_source_manifest, format_manifest_for_prompt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SUFFICIENCY THRESHOLDS
# ─────────────────────────────────────────────────────────────
# Minimum word count from internal sources before we consider the context
# sufficient and skip the web search tier.
RAG_SUFFICIENT_MIN_WORDS = 80

# Phrases that signal the RAG result is empty or a refusal, even if non-empty.
_RAG_EMPTY_SIGNALS = ("no supplemental", "no relevant", "not found", "no result")


# ─────────────────────────────────────────────────────────────
# IMPOSSIBLE SCENARIO PATTERNS
# These are caught BEFORE the LLM — instant hard refusal.
# ─────────────────────────────────────────────────────────────
IMPOSSIBLE_PATTERNS = [
    (r"\b(in |on |for |to )?(outer )?space\b",
     "Crop farming is NOT POSSIBLE in outer space. "
     "There is no atmosphere, gravity, or natural soil required for conventional agriculture."),

    (r"\b(on |for |to )?(the )?moon\b",
     "Crop farming is NOT POSSIBLE on the moon. "
     "The moon has no atmosphere, no liquid water, and temperatures incompatible with any crop."),

    (r"\b(on |for |to )?(the )?mars\b",
     "Conventional crop farming is NOT FEASIBLE on Mars. "
     "Mars has toxic perchlorates in its soil, near-zero atmospheric pressure, and no liquid water."),

    (r"\b(under|beneath|below) (the )?(sea|ocean|water)\b",
     "Crop farming underwater is NOT POSSIBLE. "
     "Crops cannot perform photosynthesis or grow in submerged saltwater conditions."),

    (r"\bph\s*(of\s*)?[01](\.\d+)?\b",
     "A soil pH of 0 to 1 is NOT FEASIBLE for any crop. "
     "No crop can survive this level of acidity. The minimum viable pH for any crop is about 3.5."),

    (r"\b(temperature|temp)\s*(of\s*)?\d{2,}\s*(degree|celsius|°C)?\b",
     "Check your temperature value. Any sustained temperature above 50°C is lethal for all crops."),

    (r"\bno (water|rain(fall)?|irrigation)\b",
     "Growing crops with zero water is NOT POSSIBLE. "
     "Even the most drought-tolerant crops need at least 200mm of annual rainfall."),
]


def _check_impossible(query: str):
    """
    Returns a firm refusal string if the query matches a known impossible scenario.
    Returns None if the query is valid.
    """
    q = query.lower()
    for pattern, reason in IMPOSSIBLE_PATTERNS:
        if re.search(pattern, q):
            return (
                f"Not Possible\n\n"
                f"{reason}\n\n"
                f"Please ask about real-world agricultural conditions such as a specific "
                f"region, season, soil parameters (pH, NPK), or a crop problem."
            )
    return None


def _assess_rag_sufficiency(rag_text: str, soil_report_context: str) -> bool:
    """
    Decide whether the internal context (document + RAG) is sufficient
    to answer the query without triggering a web search.

    Returns True  → internal context is sufficient, skip web search.
    Returns False → internal context is thin; fall through to Tier 3.
    """
    combined = " ".join(filter(None, [soil_report_context or "", rag_text or ""]))
    if not combined.strip():
        return False

    word_count = len(combined.split())
    lower = combined.lower()

    # Explicit empty-signal strings mean the KB returned nothing useful
    if any(sig in lower for sig in _RAG_EMPTY_SIGNALS):
        return False

    return word_count >= RAG_SUFFICIENT_MIN_WORDS


def _extract_season(query: str) -> str:
    """Extract an explicit season or month from the query string."""
    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|october|november|december|summer|kharif|rabi|winter|monsoon)\b",
        query.lower()
    )
    return m.group(1) if m else ""


def _wrap(message: str, header: str) -> str:
    """Wrap a short message in the standard bordered terminal frame."""
    d = "─" * 60
    return f"\n{d}\n{header}\n{d}\n\n{message}\n\n{d}\n"


def extract_follow_ups(text: str) -> list:
    """Extract follow-up questions from the LLM response."""
    questions = []
    pattern = r"\*\*Follow-up Questions\*\*(.*?)(?:\n\*\*|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
        questions = re.findall(r"(?:^|\n)[-*•\d\.]+\s*(.+)", content)
        questions = [q.strip() for q in questions if q.strip()]

    if not questions:
        questions = [q.strip() for q in re.findall(r"([^.?!\n]+\?)", text) if len(q) > 10]

    return questions[:3]


def clean_response(text: str) -> str:
    """Remove ONLY the follow-up questions section from the main text for cleaner UI."""
    pattern = r"\*\*Follow-up Questions\*\*[\s\S]*?(?=\n\*\*[^*]+\*\*|$)"
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n---\s*$", "", cleaned.strip())
    return cleaned.strip()


# ─────────────────────────────────────────────────────────────
# HIERARCHICAL KNOWLEDGE RETRIEVAL
# ─────────────────────────────────────────────────────────────

def _fetch_context_hierarchical(
    query: str,
    soil_report_context: str,
    trace: dict,
    log_step,
) -> tuple[str, dict]:
    """
    Implements the three-tier retrieval hierarchy.

    Returns:
        full_context (str)      — formatted context string ready for LLM injection
        retrieval_meta (dict)   — per-tier results for the source manifest
    """
    retrieval_meta = {
        "tier1_document": {"used": False, "text": ""},
        "tier2_knowledge_base": {"used": False, "text": ""},
        "tier3_web_search": {"used": False, "text": "", "triggered_reason": ""},
    }

    context_parts = []

    # ── Tier 1: Uploaded Document ─────────────────────────────
    log_step("Tier 1 — Checking uploaded document...")
    if soil_report_context and soil_report_context.strip():
        retrieval_meta["tier1_document"]["used"] = True
        retrieval_meta["tier1_document"]["text"] = soil_report_context.strip()
        context_parts.append(
            "[SOURCE: Uploaded Document — Tier 1, Highest Trust]\n"
            f"{soil_report_context.strip()}"
        )
        logger.info("Retrieval Tier 1 (Uploaded Document): FOUND")
    else:
        logger.info("Retrieval Tier 1 (Uploaded Document): NOT AVAILABLE")

    # ── Tier 2: Internal Knowledge Base (Vector DB) ───────────
    log_step("Tier 2 — Searching internal knowledge base...")
    rag_text = ""
    try:
        rag_text = retrieve(query.strip(), k=5)[0]
    except Exception as exc:
        logger.warning(f"Retrieval Tier 2 (Knowledge Base) error: {exc}")
        rag_text = ""

    rag_useful = (
        bool(rag_text.strip())
        and not any(sig in rag_text.lower() for sig in _RAG_EMPTY_SIGNALS)
    )

    if rag_useful:
        retrieval_meta["tier2_knowledge_base"]["used"] = True
        retrieval_meta["tier2_knowledge_base"]["text"] = rag_text.strip()
        context_parts.append(
            "[SOURCE: Internal Knowledge Base — Tier 2, High Trust]\n"
            f"{rag_text.strip()}"
        )
        logger.info("Retrieval Tier 2 (Knowledge Base): FOUND")
    else:
        logger.info("Retrieval Tier 2 (Knowledge Base): INSUFFICIENT")

    # ── Tier 3: Web Search (conditional fallback) ─────────────
    internal_sufficient = _assess_rag_sufficiency(rag_text, soil_report_context)

    if not internal_sufficient:
        reason = (
            "Uploaded document not present and knowledge base returned insufficient context."
            if not retrieval_meta["tier1_document"]["used"] and not rag_useful
            else "Knowledge base context below sufficiency threshold — supplementing with web search."
        )
        log_step("Tier 3 — Internal sources insufficient; performing web search...")
        logger.info(f"Retrieval Tier 3 (Web Search): TRIGGERED — {reason}")

        web_text = ""
        try:
            web_text = web_agent(query)
        except Exception as exc:
            logger.warning(f"Retrieval Tier 3 (Web Search) error: {exc}")
            web_text = ""

        web_useful = (
            bool(web_text.strip())
            and "No relevant" not in web_text
        )

        if web_useful:
            retrieval_meta["tier3_web_search"]["used"] = True
            retrieval_meta["tier3_web_search"]["text"] = web_text.strip()
            retrieval_meta["tier3_web_search"]["triggered_reason"] = reason
            context_parts.append(
                "[SOURCE: Web Search — Tier 3, Medium Trust]\n"
                f"{web_text.strip()}"
            )
            logger.info("Retrieval Tier 3 (Web Search): FOUND")
        else:
            logger.info("Retrieval Tier 3 (Web Search): NO RESULTS")
    else:
        logger.info("Retrieval Tier 3 (Web Search): SKIPPED — internal sources sufficient")
        retrieval_meta["tier3_web_search"]["triggered_reason"] = "Skipped — internal sources sufficient."

    full_context = "\n\n".join(context_parts) if context_parts else "No context available from any source."

    # Store results in trace
    trace["context"]["rag"] = rag_text
    trace["context"]["web"] = retrieval_meta["tier3_web_search"]["text"]
    trace["context"]["retrieval_meta"] = retrieval_meta

    return full_context, retrieval_meta


# ─────────────────────────────────────────────────────────────
# SOURCE ATTRIBUTION BLOCK (injected into LLM prompt)
# ─────────────────────────────────────────────────────────────

def _build_attribution_block(retrieval_meta: dict) -> str:
    """
    Produce a concise source attribution block for the LLM prompt.
    This tells the model exactly what was retrieved from where,
    so it can attribute claims correctly in its response.
    """
    lines = [
        "─────────────────────────────────────────",
        "KNOWLEDGE SOURCE ATTRIBUTION",
        "─────────────────────────────────────────",
    ]

    t1 = retrieval_meta["tier1_document"]
    t2 = retrieval_meta["tier2_knowledge_base"]
    t3 = retrieval_meta["tier3_web_search"]

    lines.append(
        f"Tier 1 — Uploaded Document:   {'AVAILABLE (cite as \"Per your soil report:\")' if t1['used'] else 'NOT AVAILABLE'}"
    )
    lines.append(
        f"Tier 2 — Knowledge Base:       {'AVAILABLE (cite as \"Per the knowledge base:\")' if t2['used'] else 'NOT AVAILABLE'}"
    )

    if t3["used"]:
        lines.append("Tier 3 — Web Search:           AVAILABLE (cite as \"Per web sources:\")")
    elif "Skipped" in t3.get("triggered_reason", ""):
        lines.append("Tier 3 — Web Search:           NOT USED (internal sources were sufficient)")
    else:
        lines.append("Tier 3 — Web Search:           NOT AVAILABLE")

    active = [
        "Tier 1 (Uploaded Document)" if t1["used"] else None,
        "Tier 2 (Knowledge Base)" if t2["used"] else None,
        "Tier 3 (Web Search)" if t3["used"] else None,
    ]
    active = [s for s in active if s]
    primary = active[0] if active else "Model general knowledge"
    lines.append(f"Primary source for this response: {primary}")
    lines.append("─────────────────────────────────────────")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_query(query: str, history: list = None, status_callback=None, include_trace=False, soil_report_context: str = None) -> dict | str:
    """
    Main pipeline entry point.
    Returns a string for display, OR a dict if include_trace is True.
    """
    trace = {
        "query": query,
        "steps": [],
        "context": {},
        "validated_crops": []
    }

    def log_step(name, data=None):
        trace["steps"].append({"step": name, "data": data})
        if status_callback: status_callback(name)

    # ── 1. Input Validation ───────────────────────────────────
    log_step("Validating input...")
    err = validate_input(query)
    if err:
        return _wrap(err, "Input Error") if not include_trace else {"response": err, "error": err, "trace": trace}

    # ── 2. Domain Guardrail ───────────────────────────────────
    log_step("Checking domain guardrails...")
    allowed, msg = guardrail_response(query, history=history)
    if not allowed:
        return _wrap(msg, "Out of Domain") if not include_trace else {"response": msg, "error": msg, "trace": trace}

    # ── 3. Hard Impossible-Scenario Check ────────────────────
    impossible = _check_impossible(query)
    if impossible:
        return _wrap(impossible, "Not Feasible") if not include_trace else {"response": impossible, "error": impossible, "trace": trace}

    # ── 4. Extract Constraints ────────────────────────────────
    log_step("Extracting constraints...")
    entities = extract_entities(query)
    region = entities.get("location") or ""
    soil_values = entities.get("soil_values", {})
    season = _extract_season(query)
    trace["context"]["entities"] = entities
    trace["context"]["season"] = season

    # ── 5. Rules Engine — Crop Filter ────────────────────────
    log_step("Filtering compatible crops...")
    validated_crops = []
    if season or soil_values:
        if soil_values:
            validated_crops = prioritize_crops(
                hard_filter_crops(soil_values),
                location=region
            )
        if not validated_crops:
            db_crops = filter_crops(
                region=region,
                month=season,
                ph=soil_values.get("pH_Value")
            )
            validated_crops = [c["name"] for c in prioritize(db_crops)[:6]]

    trace["validated_crops"] = validated_crops

    # ── 6. Hierarchical Knowledge Retrieval ──────────────────
    full_context, retrieval_meta = _fetch_context_hierarchical(
        query=query,
        soil_report_context=soil_report_context,
        trace=trace,
        log_step=log_step,
    )

    # ── 7. Build Source Attribution + Source Manifest ─────────
    attribution_block = _build_attribution_block(retrieval_meta)

    source_manifest = build_source_manifest(
        query=query,
        rag_text=retrieval_meta["tier2_knowledge_base"]["text"],
        web_text=retrieval_meta["tier3_web_search"]["text"],
        soil_report_context=soil_report_context,
        history=history,
    )
    manifest_block = format_manifest_for_prompt(source_manifest)
    trace["context"]["source_manifest"] = source_manifest

    # ── 8. Build the LLM Prompt ───────────────────────────────
    constraint_info = ""
    if region or season:
        constraint_info = f"\nDetected Location/Season: Location={region or 'Any'}, Season={season or 'Any'}"

    llm_query = (
        f"{attribution_block}\n\n"
        f"User Query: {query}{constraint_info}\n\n"
        f"Instructions:\n"
        f"- Structure your response as: Summary, Analysis, Recommendations, "
        f"Assumptions and Limitations (if any), Next Steps.\n"
        f"- In the Analysis section, cite each claim with its source tier using plain "
        f"attribution: 'Per your soil report:' (Tier 1), 'Per the knowledge base:' (Tier 2), "
        f"or 'Per web sources:' (Tier 3). If none of the tiers provided the information, "
        f"state 'Based on general agronomic knowledge:' and note it as an assumption.\n"
        f"- Do not fabricate data. If a specific value is unavailable from any source, "
        f"say so explicitly and provide the safest general guidance instead.\n"
        f"- If critical data is missing (field size, crop variety, NPK values), "
        f"state what is missing and ask once, clearly.\n"
        f"- Include an Advanced Details section only if the query is explicitly technical "
        f"or the user requested it.\n"
        f"- Be direct and concise. No decorative emoji as section headers."
    )

    # ── 9. LLM Synthesis with Graceful Fallback ───────────────
    try:
        log_step("Synthesizing expert response...")
        response = generate_response(query=llm_query, context=full_context, history=history)
    except Exception as e:
        logger.error(f"LLM Synthesis failed: {str(e)}")
        rag_text = retrieval_meta["tier2_knowledge_base"]["text"]
        if rag_text:
            response = (
                "**Limited Capability Mode**\n\n"
                "The analysis engine is temporarily unavailable. "
                "The following data was retrieved from the internal knowledge base:\n\n"
                f"{rag_text[:1000]}"
                "\n\n*Please retry for a full expert analysis.*"
            )
        else:
            response = (
                "**Service Unavailable**: Unable to connect to the analysis engine "
                "or retrieve local data. Please check your internet connection and try again."
            )

    follow_ups = extract_follow_ups(response)
    cleaned_answer = clean_response(response)

    if include_trace:
        return {
            "response": cleaned_answer,
            "follow_ups": follow_ups,
            "full_raw": response,
            "trace": trace
        }

    return cleaned_answer
