"""
AgriSense-AI — utils/context_classifier.py
Builds a structured source manifest from all available inputs.

This manifest is injected into the LLM prompt so the model explicitly knows:
  - What data came from where
  - What confidence level each source carries
  - What critical information is missing (preventing definitive answers)

This powers the Evidence Classification block (✅ FACT / 🔵 DOMAIN KNOWLEDGE /
🟡 ASSUMPTION / ❓ MISSING INFO) in every AI response.
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────
# CONFIDENCE LEVELS
# ─────────────────────────────────────────────────────────────
CONFIDENCE_HIGH   = "HIGH"    # Direct user-provided or lab-verified data
CONFIDENCE_MEDIUM = "MEDIUM"  # Retrieved / scraped — trusted but indirect
CONFIDENCE_LOW    = "LOW"     # Inferred / assumed — should be flagged


# ─────────────────────────────────────────────────────────────
# CRITICAL AGRI PARAMETERS — if absent, the AI must flag them
# ─────────────────────────────────────────────────────────────
CRITICAL_PARAMS = {
    "field_size":      r"\b(\d+(\.\d+)?\s*(acre|hectare|bigha|guntha|cent|katha|marla)s?)\b",
    "soil_ph":         r"\bph\s*(of\s*|is\s*|=\s*|:)?\s*([3-9](\.\d{1,2})?)\b",
    "crop_variety":    r"\b(variety|cultivar|hybrid|strain|type)\b",
    "nitrogen":        r"\b(nitrogen|urea|N\s*[:=]?\s*\d+)\b",
    "phosphorus":      r"\b(phosphorus|phosphate|DAP|P\s*[:=]?\s*\d+)\b",
    "potassium":       r"\b(potassium|potash|MOP|K\s*[:=]?\s*\d+)\b",
    "location":        r"\b(district|state|region|zone|village|taluk|block|mandal)\b",
    "season":          r"\b(kharif|rabi|zaid|summer|winter|monsoon|june|july|october|november)\b",
    "irrigation_type": r"\b(drip|sprinkler|flood|furrow|rainfed|irrigated)\b",
    "growth_stage":    r"\b(seedling|vegetative|tillering|flowering|fruiting|maturity|harvest)\b",
}


def _detect_entities_in_text(text: str) -> dict:
    """Scan text for presence of critical agri parameters."""
    found = {}
    if not text:
        return found
    t = text.lower()
    for param, pattern in CRITICAL_PARAMS.items():
        match = re.search(pattern, t, re.IGNORECASE)
        if match:
            found[param] = match.group(0).strip()
    return found


def _summarize_history(history: list) -> dict:
    """Extract key stats and entities from conversation history."""
    if not history:
        return {"turns": 0, "key_entities": [], "summary": "No prior conversation."}

    turns = len([m for m in history if m.get("role") == "user"])
    all_text = " ".join(m.get("content", "") for m in history)
    entities_found = _detect_entities_in_text(all_text)

    entity_list = []
    for key, val in entities_found.items():
        entity_list.append(f"{key.replace('_', ' ').title()}: {val}")

    return {
        "turns": turns,
        "key_entities": entity_list[:8],  # cap at 8 for prompt brevity
        "summary": (
            f"{turns} prior user turn(s). "
            f"Entities remembered: {', '.join(entity_list[:5]) if entity_list else 'none detected'}."
        )
    }


def _assess_document_quality(doc_text: str) -> dict:
    """Assess the richness of an uploaded soil/document context."""
    if not doc_text or not doc_text.strip():
        return {"available": False, "confidence": None, "summary": "No document uploaded."}

    entities = _detect_entities_in_text(doc_text)
    word_count = len(doc_text.split())

    # Determine confidence based on richness
    if len(entities) >= 4 or word_count > 200:
        confidence = CONFIDENCE_HIGH
        quality_note = "Rich document — multiple parameters detected."
    elif len(entities) >= 2 or word_count > 50:
        confidence = CONFIDENCE_MEDIUM
        quality_note = "Partial document — some parameters detected."
    else:
        confidence = CONFIDENCE_LOW
        quality_note = "Sparse document — few parameters detected; treat with caution."

    return {
        "available": True,
        "confidence": confidence,
        "entities_detected": entities,
        "word_count": word_count,
        "quality_note": quality_note,
        "summary": doc_text[:300].strip() + ("..." if len(doc_text) > 300 else ""),
    }


def _detect_missing_critical(
    query: str,
    doc_entities: dict,
    history_entities: list,
) -> list:
    """
    Identify critical parameters that are absent from ALL sources.
    These must be flagged as ❓ MISSING INFO in the AI response.
    """
    query_entities = _detect_entities_in_text(query)
    history_text = " ".join(history_entities)

    all_known_params = set(query_entities.keys()) | set(doc_entities.keys())
    # Check if history mentions the param key word
    for param in CRITICAL_PARAMS:
        if any(param.replace("_", " ") in e.lower() for e in history_entities):
            all_known_params.add(param)

    # Determine which critical params are absent
    missing = []
    essential_always = ["field_size", "location"]
    essential_for_fertilizer = ["soil_ph", "nitrogen", "phosphorus", "potassium"]

    q_lower = query.lower()
    needs_fertilizer = any(
        kw in q_lower for kw in ["fertilizer", "fertiliser", "npk", "urea", "dose", "dosage", "nutrient"]
    )
    needs_location = any(
        kw in q_lower for kw in ["recommend", "suggest", "which crop", "suitable", "best crop", "grow"]
    )

    for param in essential_always:
        if needs_location and param not in all_known_params:
            missing.append(param.replace("_", " ").replace("-", " ").title())

    if needs_fertilizer:
        for param in essential_for_fertilizer:
            if param not in all_known_params:
                missing.append(param.replace("_", " ").replace("-", " ").title())

    return list(dict.fromkeys(missing))  # deduplicated, ordered


def build_source_manifest(
    query: str,
    rag_text: str = "",
    web_text: str = "",
    soil_report_context: Optional[str] = None,
    history: Optional[list] = None,
) -> dict:
    """
    Build a comprehensive source manifest for a single query turn.

    Returns a dict with:
      - sources: per-source availability, confidence, and summary
      - data_quality: overall quality ("HIGH" | "MEDIUM" | "LOW")
      - missing_critical: list of missing critical parameters
      - prompt_block: a pre-formatted string ready to inject into the LLM prompt
    """
    doc_info      = _assess_document_quality(soil_report_context)
    history_info  = _summarize_history(history or [])
    has_rag       = bool(rag_text and rag_text.strip() and "No supplemental" not in rag_text)
    has_web       = bool(web_text and web_text.strip() and "No relevant" not in web_text)

    sources = {
        "uploaded_document": doc_info,
        "rag_knowledge_base": {
            "available":   has_rag,
            "confidence":  CONFIDENCE_MEDIUM if has_rag else None,
            "summary":     (rag_text[:200].strip() + "...") if has_rag else "No local knowledge retrieved.",
        },
        "web_search": {
            "available":   has_web,
            "confidence":  CONFIDENCE_MEDIUM if has_web else None,
            "summary":     (web_text[:200].strip() + "...") if has_web else "No web results available.",
        },
        "conversation_history": history_info,
    }

    # Overall data quality
    has_high = doc_info.get("confidence") == CONFIDENCE_HIGH
    has_any  = any([
        doc_info.get("available"),
        has_rag,
        has_web,
        history_info["turns"] > 0,
    ])
    if has_high:
        data_quality = CONFIDENCE_HIGH
    elif has_any:
        data_quality = CONFIDENCE_MEDIUM
    else:
        data_quality = CONFIDENCE_LOW

    # Detect missing critical information
    doc_entities = doc_info.get("entities_detected", {}) if doc_info.get("available") else {}
    missing_critical = _detect_missing_critical(
        query=query,
        doc_entities=doc_entities,
        history_entities=history_info.get("key_entities", []),
    )

    # ── Build the human-readable prompt block ────────────────
    lines = [
        "═══════════════════════════════════════════════════",
        "📂 AVAILABLE CONTEXT MANIFEST (use for Evidence Classification)",
        "═══════════════════════════════════════════════════",
    ]

    # Uploaded document
    if doc_info["available"]:
        lines.append(
            f"📄 UPLOADED DOCUMENT  [{doc_info['confidence']} CONFIDENCE]"
            f"\n   Quality: {doc_info.get('quality_note', '')}"
            f"\n   Detected parameters: {', '.join(doc_info.get('entities_detected', {}).keys()) or 'none'}"
            f"\n   Preview: {doc_info['summary'][:200]}"
        )
    else:
        lines.append("📄 UPLOADED DOCUMENT  [NOT AVAILABLE] — No soil report uploaded for this session.")

    # RAG
    if has_rag:
        lines.append(f"📚 LOCAL KNOWLEDGE BASE  [{CONFIDENCE_MEDIUM} CONFIDENCE]\n   Preview: {rag_text[:150].strip()}...")
    else:
        lines.append("📚 LOCAL KNOWLEDGE BASE  [NOT AVAILABLE]")

    # Web
    if has_web:
        lines.append(f"🌐 WEB SEARCH RESULTS  [{CONFIDENCE_MEDIUM} CONFIDENCE]\n   Preview: {web_text[:150].strip()}...")
    else:
        lines.append("🌐 WEB SEARCH RESULTS  [NOT AVAILABLE]")

    # History
    lines.append(
        f"🕐 CONVERSATION HISTORY  [{history_info['turns']} prior user turn(s)]"
        f"\n   Remembered entities: {', '.join(history_info['key_entities']) or 'none'}"
    )

    # Missing info
    if missing_critical:
        lines.append(
            "❓ MISSING CRITICAL PARAMETERS (flag these as ❓ MISSING INFO in your response):"
            f"\n   {', '.join(missing_critical)}"
        )
    else:
        lines.append("✅ All critical parameters present or resolvable from context.")

    lines.append(f"📊 OVERALL DATA QUALITY: {data_quality}")
    lines.append("═══════════════════════════════════════════════════")

    prompt_block = "\n".join(lines)

    return {
        "sources":          sources,
        "data_quality":     data_quality,
        "missing_critical": missing_critical,
        "prompt_block":     prompt_block,
    }


def format_manifest_for_prompt(manifest: dict) -> str:
    """Return the pre-built prompt block string from a manifest."""
    return manifest.get("prompt_block", "")
