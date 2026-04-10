"""
AgriSense-AI — utils/formatter.py
A single, clean formatter for all responses.
"""


def format_response(answer: str, confidence: str = "High", intent: str = "general", query: str = "") -> str:
    """
    Wraps a raw LLM answer in a clean, readable terminal block.
    Used when the pipeline returns a general knowledge answer (no crop list).
    """
    badge = {
        "High": "🟢",
        "Medium": "🟡",
        "Low": "🔴",
        "RAG Knowledge": "🟢",
    }.get(confidence, "🟡")

    divider = "─" * 60

    header = {
        "decision":   "🌱 Crop Recommendation",
        "comparison": "⚖️  Crop Comparison",
        "knowledge":  "📘 Agricultural Knowledge",
        "procedure":  "📋 How-To Guide",
        "diagnosis":  "🔍 Pest / Disease Diagnosis",
        "fertilizer": "🧪 Fertilizer Guidance",
        "location":   "📍 Region-Specific Information",
        "general":    "📘 AgriSense Answer",
    }.get(intent, "📘 AgriSense Answer")

    return (
        f"\n{divider}\n"
        f"{header}\n"
        f"{divider}\n\n"
        f"{answer}\n\n"
        f"{divider}\n"
        f"{badge} Confidence: {confidence}\n"
    )


def format_output(crops: list, explanation: str) -> str:
    """
    Formats the deterministic crop recommendation output.
    Used when the pipeline has produced a validated crop list.
    """
    divider = "─" * 60
    crop_bullets = "\n".join(f"  • {c}" for c in crops)

    return (
        f"\n{divider}\n"
        f"🌱 Crop Recommendation\n"
        f"{divider}\n\n"
        f"Validated suitable crops:\n{crop_bullets}\n\n"
        f"{explanation}\n\n"
        f"{divider}\n"
    )


def format_crop_result(top_list: list, explanation: str,
                       soil_summary: str = "", confidence: str = "Medium",
                       used_defaults: list = None) -> str:
    """Deprecated stub — ML model has been removed."""
    crops = [item["crop"] if isinstance(item, dict) else str(item) for item in top_list]
    return format_output(crops, explanation)
