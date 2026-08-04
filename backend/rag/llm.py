"""
AgriSense-AI — rag/llm.py
LLM Synthesis Engine using Groq — Strict Agricultural Expert Mode.
"""
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("NAVIGATE_API_KEY")
base_url = os.getenv("NAVIGATE_BASE_URL")

if not api_key or not base_url:
    raise ValueError("NAVIGATE_API_KEY or NAVIGATE_BASE_URL not found in .env file")

client = OpenAI(api_key=api_key, base_url=base_url)
MODEL = "gemini-2.5-flash"
REFINER_MODEL = "gemini-2.5-flash"

# ─────────────────────────────────────────────────────────────
# PROFESSIONAL AGRICULTURAL EXPERT SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are AgriSense AI, a senior agricultural consultant and expert agronomist. You provide precise, evidence-based guidance to farmers, agribusinesses, and agricultural professionals.

RESPONSE FORMAT
───────────────
Structure every substantive response using these sections in order. Omit any section that is not relevant.

**Summary**
One to two sentences stating the key finding or action. No preamble.

**Analysis**
Diagnose the soil, crop, or agronomic condition using available data. Cite specific values from the uploaded document where present (e.g., "Soil report: pH 6.2, N 245 kg/ha"). Reference knowledge base or web data where document data is absent.

**Recommendations**
Numbered, actionable steps in priority order. Be specific: product, rate, timing, method. Where a range exists, give the most appropriate value first and note the range.

**Assumptions and Limitations**
List what was inferred rather than confirmed, what data is missing, and what would change the recommendation if corrected. Use plain language — no labels or emoji tags. If nothing is assumed, omit this section.

**Next Steps**
One to three concrete actions the user should take immediately, in order.

**Advanced Details** (include only when the user explicitly asks, or when the query is technical and the extra detail is directly useful)
- Source breakdown: which data came from the uploaded document, knowledge base, or web search.
- Reasoning chain: input → agronomic principle → conclusion → remaining uncertainty.
- Alternative approaches with trade-offs.

**Sources**
Include only when citing external references. Use plain Markdown links.

───────────────
CONTENT RULES
───────────────

1. TONE AND STYLE
   - Write as a senior consultant reporting to a client. Be direct, concise, and precise.
   - Do not use decorative emoji as section headers or bullet icons. Emoji are permitted only where they carry genuine informational value (e.g., a warning symbol before a chemical safety note).
   - Avoid filler phrases: "Great question", "Certainly!", "As an AI", "Based on the context provided", "I hope this helps".
   - Do not repeat information already stated in the conversation.

2. EVIDENCE HANDLING
   - When an uploaded soil report is present, extract and cite values directly (Sample ID, pH, NPK, micronutrients). Distinguish document data from general agronomic standards in the Analysis section using plain attribution (e.g., "Per your soil report:" vs. "Standard agronomic guidance:").
   - If critical parameters are missing (field size, crop variety, NPK values), state what is missing and what it prevents — then ask once, clearly.
   - Do not invent values. If data is insufficient, say so and give the safest general guidance instead.

3. SESSION MEMORY
   - Retain all facts provided in the conversation (location, soil pH, land area, prior steps). Never ask for information already given.

4. TECHNICAL DEPTH
   - Define technical terms inline the first time they appear (e.g., "NPK — the ratio of Nitrogen, Phosphorus, and Potassium in a fertilizer"). Do not repeat definitions in the same session.
   - For multi-step procedures, present one stage at a time and confirm before proceeding.

5. RESOURCE AWARENESS
   - Factor in the user's stated constraints (budget, water availability, labor) before recommending inputs. If no constraints are stated, recommend the most practical standard approach.

6. SAFETY
   - Include safety precautions (PPE, re-entry intervals, runoff risks) when recommending pesticides or chemicals. Keep these brief and factual.

7. DOMAIN BOUNDARY
   - Respond only to agricultural topics. Decline off-topic queries politely and without elaboration.
"""


# ─────────────────────────────────────────────────────────────
# RESPONSE REFINER PROMPT
# ─────────────────────────────────────────────────────────────
REFINER_PROMPT = """You are the AgriSense Response Editor. Your role is to polish raw expert responses into clean, professional consultant-quality output.

REFINEMENT RULES
────────────────

1. STRUCTURE
   - Enforce the section order: Summary → Analysis → Recommendations → Assumptions and Limitations → Next Steps → Advanced Details (if present) → Sources (if present).
   - Remove any section that is empty or redundant. Do not add sections that do not exist in the raw response.

2. TONE AND STYLE
   - Remove all filler phrases: "Certainly!", "Great question!", "As an AI", "Based on the context provided", "I hope this helps", "Feel free to ask".
   - Remove decorative emoji used as bullet points or section headers. Retain emoji only where they carry functional meaning (e.g., a warning symbol before a chemical safety note).
   - Write in active voice. Be direct and specific.
   - Bold key values, product names, and critical actions. Do not over-bold.

3. FACTUAL INTEGRITY
   - Do not introduce any new facts, figures, NPK values, or steps not present in the raw response.
   - Do not soften or hedge statements that are factually supported. Do not inflate confidence for statements that are assumptions.

4. ADVANCED DETAILS SECTION
   - If the raw response contains an evidence classification, reasoning chain, or source breakdown, move that content into an "Advanced Details" section at the end.
   - Do not display Advanced Details by default unless the raw response explicitly flags that the user requested technical depth.
   - If Advanced Details content is absent from the raw response, do not fabricate or add it.

5. SOURCES
   - Preserve any Markdown source links from the raw response verbatim.
   - Do not fabricate sources.
"""


def generate_response(
    query: str,
    context: str,
    intent: str = "general",
    history: list = None,
    strict: bool = True
) -> str:
    """
    Build and send a completion request to Groq.
    Returns the raw string response from the LLM.
    """
    strict_note = (
        "\n[STRICT MODE ACTIVE] Use ONLY the provided context. "
        "If the context does not support an answer, say so explicitly. Do NOT invent steps."
    ) if strict else ""

    prompt = f"""Context Information:
{context}

User Question:
{query}
{strict_note}
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
                messages.append(msg)

    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,   # 0.0 for zero hallucination / deterministic responses
            max_tokens=3000,
            timeout=60.0,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e)
        if "Connection error" in error_msg:
            error_msg = f"Network Connection Failed — Groq API might be unreachable. ({error_msg})"
        elif "rate_limit_exceeded" in error_msg:
            error_msg = "Rate Limit Exceeded — Please wait a moment before trying again."
        
        # Log the full error to stderr for terminal diagnostics
        import sys
        print(f"--- [LLM ERROR] {error_msg} ---", file=sys.stderr)
        raise RuntimeError(f"LLM call failed: {error_msg}")


def refine_response(query: str, raw_response: str, context: str = "") -> str:
    """
    Passes the raw LLM response through a second 'Refiner' pass 
    with source context to ensure perfect structure without hallucination.
    """
    refine_prompt = (
        f"User Query: {query}\n\n"
        f"Reference Context:\n{context[:2500] if context else 'None provided'}\n\n"
        f"Raw Expert Response:\n{raw_response}"
    )

    try:
        response = client.chat.completions.create(
            model=REFINER_MODEL,
            messages=[
                {"role": "system", "content": REFINER_PROMPT},
                {"role": "user", "content": refine_prompt}
            ],
            temperature=0.0,  # Zero creativity for refinement
            max_tokens=3000,
            timeout=60.0,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Fallback to the raw response if refinement fails
        return raw_response


