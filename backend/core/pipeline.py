"""
AgriSense-AI — core/pipeline.py
The single authoritative entry point for all queries.

Flow:
  Query → Validate → Guardrail → Impossible-Check → Extract Constraints
       → Filter Crops (Rules Engine) → Fetch Context → LLM → Format Output
"""

import re
import logging
from core.crop_engine import filter_crops, prioritize
from utils.validator import validate_input, extract_entities, hard_filter_crops, prioritize_crops
from utils.guardrails import guardrail_response
from rag.llm import generate_response, refine_response
from rag.rag_pipeline import retrieve
from agents.web_agent import web_agent

logger = logging.getLogger(__name__)


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
                f"❌ NOT POSSIBLE\n\n"
                f"{reason}\n\n"
                f"Please ask about real-world agricultural conditions such as:\n"
                f"  • A specific region or state (e.g., Punjab, Kerala)\n"
                f"  • A season (summer, kharif, rabi, winter)\n"
                f"  • Soil parameters (pH, NPK, rainfall)\n"
                f"  • A crop problem (disease, pest, yield)"
            )
    return None


def _fetch_context(query: str) -> str:
    """Retrieve RAG + web knowledge context."""
    try:
        rag_text, _ = retrieve(f"{query} agriculture", k=5)
    except Exception:
        rag_text = ""

    web_text = web_agent(query)

    parts = []
    if rag_text and rag_text.strip():
        parts.append(f"[Knowledge Base]\n{rag_text.strip()}")
    if web_text and web_text.strip() and "No relevant" not in web_text:
        parts.append(f"[Web Search Results]\n{web_text.strip()}")

    return "\n\n".join(parts) if parts else "No supplemental context found."


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
    # Look for the section with any variant of "Follow-up Questions"
    pattern = r"\*\*Follow-up Questions\*\*(.*?)(?:\n\*\*|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
        # Extract lines starting with bullet points or dashes
        questions = re.findall(r"(?:^|\n)[-*•\d\.]+\s*(.+)", content)
        questions = [q.strip() for q in questions if q.strip()]
    
    # Fallback: look for lines ending in ? if no section found
    if not questions:
        questions = [q.strip() for q in re.findall(r"([^.?!\n]+\?)", text) if len(q) > 10]

    return questions[:3]


def clean_response(text: str) -> str:
    """Remove the follow-up questions section from the main text for cleaner UI."""
    cleaned = re.sub(r"\*\*Follow-up Questions\*\*.*?(?:\n\*\*|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove trailing Horizontal Rules or extra whitespace
    cleaned = re.sub(r"\n---\s*$", "", cleaned.strip())
    return cleaned.strip()


def run_query(query: str, history: list = None, status_callback=None, include_trace=False) -> dict | str:
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
    log_step("🔍 Validating input...")
    err = validate_input(query)
    if err:
        return _wrap(err, "⚠️ Input Error") if not include_trace else {"error": err, "trace": trace}

    # ── 2. Domain Guardrail ───────────────────────────────────
    log_step("🛡️ Checking domain guardrails...")
    allowed, msg = guardrail_response(query)
    if not allowed:
        return _wrap(msg, "🚫 Out of Domain") if not include_trace else {"error": msg, "trace": trace}

    # ── 3. Hard Impossible-Scenario Check ────────────────────
    impossible = _check_impossible(query)
    if impossible:
        return _wrap(impossible, "❌ Not Feasible") if not include_trace else {"error": impossible, "trace": trace}

    # ── 4. Extract Constraints ────────────────────────────────
    log_step("🚜 Extracting constraints...")
    entities = extract_entities(query)
    region = entities.get("location") or ""
    soil_values = entities.get("soil_values", {})
    season = _extract_season(query)
    trace["context"]["entities"] = entities
    trace["context"]["season"] = season

    # ── 5. Rules Engine — Crop Filter ────────────────────────
    log_step("🌾 Filtering compatible crops...")
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

    # ── 6. Fetch Background Knowledge ────────────────────────
    log_step("🌐 Fetching web & local knowledge...")
    try:
        rag_text, _ = retrieve(f"{query} agriculture", k=5)
    except Exception:
        rag_text = ""
    
    web_text = web_agent(query)
    
    trace["context"]["rag"] = rag_text
    trace["context"]["web"] = web_text

    context_parts = []
    if rag_text.strip(): context_parts.append(f"[Local RAG]\n{rag_text.strip()}")
    if web_text.strip() and "No relevant" not in web_text: context_parts.append(f"[Web Search]\n{web_text.strip()}")
    full_context = "\n\n".join(context_parts) if context_parts else "No supplemental context found."

    # ── 7. Build the LLM Prompt ──────────────────────────────
    if validated_crops:
        crop_list = ", ".join(validated_crops)
        llm_query = (
            f"User Query: {query}\n"
            f"Database Validated Crops: {crop_list}\n"
            f"Detected Constraints: Location={region or 'Any'}, Season={season or 'Any'}\n\n"
            f"TASK: Provide raw agricultural data points based on the CONTEXT.\n"
            f"1. Extract scientific reasons why these crops fit the constraints.\n"
            f"2. List technical steps (Sowing, NPK, Harvest).\n"
            f"3. Identify any missing critical data (pH, exact month) for follow-up.\n"
            f"4. Do NOT output empty labels like 'Sowing:', 'NPK:', or 'Harvest:'. If specific values are missing, write 'Not specified in context — provide [items]' and include them in Follow-up Questions.\n"
            f"5. Use concise, pipe-separated steps with concrete values where available.\n"
            f"NO intro filler. Just raw, technical bullet points."
        )
    else:
        llm_query = (
            f"User Query: {query}\n\n"
            f"TASK: Extract direct agricultural answers from CONTEXT.\n"
            f"1. Give direct scientific facts. No 'Based on the context' talk.\n"
            f"2. List 2-3 specific follow-up questions if details are missing.\n"
            f"3. Do NOT leave empty labels; if data is missing, write 'Not specified in context — provide [items]'.\n"
            f"4. State 'NOT POSSIBLE' if query is scientifically unsound."
        )

    # ── 8. LLM Synthesis ─────────────────────────────────────
    log_step("🧠 Synthesizing expert response...")
    raw_response = generate_response(query=llm_query, context=full_context, history=history)

    log_step("✨ Refining expert response...")
    response = refine_response(query=query, raw_response=raw_response)

    follow_ups = extract_follow_ups(response)
    cleaned_answer = clean_response(response)

    if include_trace:
        return {
            "response": cleaned_answer, 
            "follow_ups": follow_ups,
            "full_raw": response,
            "trace": trace
        }
    
    # For backward compatibility with terminal/basic calls, still return string
    # but we can check the return type elsewhere.
    return cleaned_answer
