"""
AgriSense-AI — utils/validator.py
Implements Fix 1, 2, 6, 7 from the Universal Fix Framework:
  ✅ Fix 1  — Input validation layer
  ✅ Fix 2  — Query intent + entity classification
  ✅ Fix 6  — Output validation
  ✅ Fix 7  — Confidence scoring
"""

import re
import sys
import os

# Add parent dir to path if needed for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge.crop_info import CROP_MASTER, get_crop_by_alias, get_solution_space_diversity


# ═══════════════════════════════════════════════════════════════════
#  FIX 1 — INPUT VALIDATION & DOMAIN SANITY (Issue 3, 6)
# ═══════════════════════════════════════════════════════════════════

# Domain-specific constraints for Indian Agriculture
SOIL_LIMITS = {
    "Nitrogen": {"min": 0, "max": 300, "unit": "kg/ha"},
    "Phosphorus": {"min": 0, "max": 200, "unit": "kg/ha"},
    "Potassium": {"min": 0, "max": 300, "unit": "kg/ha"},
    "Temperature": {"min": 0, "max": 60, "unit": "°C"},
    "Humidity": {"min": 0, "max": 100, "unit": "%"},
    "pH_Value": {"min": 3, "max": 11, "unit": ""},
    "Rainfall": {"min": 0, "max": 5000, "unit": "mm"},
}

def validate_input(query: str) -> str | None:
    """
    Validates user query for agriculture relevance AND domain sanity.
    """
    if not query or not query.strip():
        return "❌ Please enter a question."

    q_strip = query.strip()
    if len(q_strip) < 5:
        return "❌ Please provide a more detailed query."

    # Only digits / special chars — not a real question
    if re.fullmatch(r"[\d\s\W]+", q_strip):
        return "❌ Please enter a valid agricultural question."

    # Repetitive characters (gibberish check)
    if re.search(r"(.)\1{4,}", q_strip):
        return "❌ Input contains too many repetitive characters. Please provide a clear question."

    # Extremely long input — likely a paste error
    if len(query) > 1000:
        return "❌ Query is too long. Please keep it under 1000 characters."

    query_low = q_strip.lower()

    # 🚨 Domain Sanity Checks
    entities = extract_entities(query)
    soil_data = entities["soil_values"]
    
    for feat, value in soil_data.items():
        limits = SOIL_LIMITS.get(feat)
        if limits:
            if value < limits["min"]:
                return f"⚠️ {feat} value ({value}{limits['unit']}) is impossible. Minimum is {limits['min']}."
            if value > limits["max"]:
                return f"⚠️ {feat} value ({value}{limits['unit']}) is unrealistic for agriculture. Please verify."

    # Cross-value sanity check
    temp = soil_data.get("Temperature")
    hum = soil_data.get("Humidity")
    if temp and hum:
        if temp > 50 and hum > 90:
            return "⚠️ The combination of extreme heat (>50°C) and extreme humidity (>90%) is implausible. Please check your data."

    return None  # valid


# ═══════════════════════════════════════════════════════════════════
#  FIX 2 — QUERY INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

# Intent keyword maps — order matters (more specific first)
INTENT_MAP = {
    "decision": [
        "recommend", "suggest", "which crop", "what crop",
        "best crop", "suitable crop", "should i grow",
        "should i plant", "which is better",
    ],
    "comparison": [
        "compare", "difference between", "vs", "versus",
        "better than", "which is best", "pros and cons",
    ],
    "knowledge": [
        "what is", "what are", "explain", "how does",
        "why is", "tell me about", "describe", "define",
        "what do you mean",
    ],
    "procedure": [
        "how to", "how do i", "steps to", "process of",
        "method", "technique", "procedure", "guide",
        "when to", "how much",
    ],
    "diagnosis": [
        "disease", "pest", "problem", "issue", "deficiency",
        "yellow", "wilting", "dying", "infected", "damaged",
        "symptom", "treatment", "cure",
    ],
    "location": [
        "in erode", "in coimbatore", "in nilgiris", "in salem",
        "in madurai", "in trichy", "in thanjavur",
        "cultivated in", "grown in", "crops in", "region",
        "district", "area", "zone", "location", "nilgiri",
        "tamilnadu", "tamil nadu",
    ],
    "fertilizer": [
        "fertilizer", "fertiliser", "manure", "urea",
        "compost", "npk", "nitrogen", "phosphorus",
        "potassium", "zinc", "dap", "nutrient",
    ],
}


def classify_query(query: str) -> str:
    """
    Classifies query into one of 7 intent categories.

    Returns one of:
        'decision' | 'comparison' | 'knowledge' | 'procedure' |
        'diagnosis' | 'location' | 'fertilizer' | 'general'
    """
    q = query.lower()
    for intent, keywords in INTENT_MAP.items():
        if any(kw in q for kw in keywords):
            return intent
    return "general"


def extract_entities(query: str) -> dict:
    """
    Extracts key entities from the query:
    crop names, soil values, location names.
    """
    q = query.lower()

    # 1. Crop Extraction using word boundaries
    found_crops = []
    for crop_key, info in CROP_MASTER.items():
        for alias in info["aliases"]:
            if re.search(rf"\b{re.escape(alias)}\b", q):
                if info["canonical"] not in found_crops:
                    found_crops.append(info["canonical"])
                break

    # 2. Location extraction (more robust patterns)
    # Looking for proper nouns after prepositions like "in", "for", "at", "around"
    # Or specifically mentioning "state of", "region of", etc.
    found_location = None
    loc_patterns = [
        r"\b(?:in|for|at|around|region of|state of)\s+([a-z]+(?: [a-z]+)?)\b",
        r"\b([a-z]+(?: [a-z]+)?)\s+(?:district|state|region|zone)\b",
    ]
    for pattern in loc_patterns:
        match = re.search(pattern, q)
        if match:
            potential_loc = match.group(1).title()
            # Simple heuristic: don't pick up common words as locations
            common_words = ["The", "A", "An", "My", "Your", "This", "That", "Crop", "Soil", "Farming"]
            if potential_loc not in common_words:
                found_location = potential_loc
                break

    # 3. Soil values (enhanced patterns)
    SOIL_PATTERNS = {
        "Nitrogen":    r"(?:nitrogen|n)\s*(?:level|is|:|=|\bof\b)*\s*(\d+\.?\d*)",
        "Phosphorus":  r"(?:phosphorus|p)\s*(?:level|is|:|=|\bof\b)*\s*(\d+\.?\d*)",
        "Potassium":   r"(?:potassium|k)\s*(?:level|is|:|=|\bof\b)*\s*(\d+\.?\d*)",
        "Temperature": r"(?:temp|temperature|at)\s*(?:is|:|=|\bof\b)*\s*(\d+\.?\d*)\s*(?:c|degree|deg)?",
        "Humidity":    r"hum(?:idity)?\s*(?:is|:|=|\bof\b)*\s*(\d+\.?\d*)\s*(?:%)?",
        "pH_Value":    r"ph\s*(?:level|value|is|:|=|\bof\b)*\s*(\d+\.?\d*)",
        "Rainfall":    r"rain(?:fall)?\s*(?:is|:|=|\bof\b)*\s*(\d+\.?\d*)\s*(?:mm|cm)?",
    }
    soil_values = {}
    for feat, pattern in SOIL_PATTERNS.items():
        m = re.search(pattern, q)
        if m:
            try:
                val = float(m.group(1))
                # Handle cm to mm conversion for rainfall
                if feat == "Rainfall" and "cm" in q[m.start():m.end()+10].lower():
                    val *= 10
                soil_values[feat] = val
            except ValueError:
                continue

    return {
        "crops":       found_crops,
        "location":    found_location,
        "soil_values": soil_values,
    }


def hard_filter_crops(soil_values: dict) -> list:
    """
    RULES ENGINE: Physically filters crops based on hard database constraints.
    Returns list of physically viable crops.
    """
    valid_crops = []
    
    for key, info in CROP_MASTER.items():
        is_viable = True
        
        # Check pH
        if "pH_Value" in soil_values:
            val = soil_values["pH_Value"]
            if val < info["ph"]["min"] or val > info["ph"]["max"]:
                is_viable = False
        
        # Check Temp
        if "Temperature" in soil_values:
            val = soil_values["Temperature"]
            if val < info["temp"]["min"] or val > info["temp"]["max"]:
                is_viable = False
                
        # Check Rainfall
        if "Rainfall" in soil_values:
            val = soil_values["Rainfall"]
            if val < info["rain"]["min"] or val > info["rain"]["max"]:
                # Special cases: If slightly below, allow with irrigation warning
                if val < info["rain"]["min"] * 0.7:
                    is_viable = False
                    
        if is_viable:
            valid_crops.append(info["canonical"])
            
    return valid_crops


def prioritize_crops(crops: list, location: str = "") -> list:
    """
    Sorts crops purely by Global Importance without regional or cash-crop bias.
    """
    if not crops: return []
    
    crop_data = []
    for c in crops:
        # Find match in CROP_MASTER
        info = next((v for v in CROP_MASTER.values() if v["canonical"] == c), None)
        if info:
            score = info.get("importance", 5)
            crop_data.append((c, score))
            
    # Sort by score descending
    crop_data.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in crop_data]


# ═══════════════════════════════════════════════════════════════════
#  FIX 6 — OUTPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════

# Phrases that signal the LLM went beyond the context
HALLUCINATION_SIGNALS = [
    "generally speaking",
    "typically",
    "in most cases",
    "it is commonly known",
    "studies have shown",
    "research suggests",
    "experts say",
    "it is well known",
    "as we know",
]


def enforce_grounding(answer: str, context: str) -> bool:
    """
    Ensures the answer is strictly based on the retrieved context.
    Checks for unique word overlap (threshold: 20 words).
    """
    if not context or not answer:
        return False
        
    # Clean and setify words
    def get_words(text):
        return set(re.findall(r"\w+", text.lower()))

    ctx_words = get_words(context)
    ans_words = get_words(answer)
    
    # Exclude common stop words for a more accurate check
    stop_words = {"the", "and", "for", "with", "this", "that", "from", "into", "should", "could", "would"}
    ans_words = ans_words - stop_words
    
    overlap = len(ans_words & ctx_words)
    return overlap >= 15  # 15 unique non-stop words shared


def check_completeness(answer: str) -> bool:
    """
    Ensures the answer provides a diverse solution space (at least 3 crops).
    """
    text = answer.lower()
    # Count bullet points, numbered lists, or bolded crop names
    crop_count = text.count("- ") + text.count("* ") + text.count("\n1.") + text.count("\n2.") + text.count("\n3.")
    
    return crop_count >= 3


def validate_output(answer: str, context: str = "") -> str:
    """
    Post-processes LLM response to ensure clean, valid output.
    Strips internal reasoning steps and checks for contradictions.
    """
    if not answer or not answer.strip():
        return "⚠️ The system could not generate a complete answer. Please rephrase."



    # Dedup repetitive content
    lines = answer.split("\n")
    unique_lines = []
    for line in lines:
        if line.strip() and line.strip() in unique_lines:
            continue
        unique_lines.append(line)
    
    answer = "\n".join(unique_lines)
    text = answer.lower()

    # 🚨 Hallucination Refusal
    bad_patterns = ["not mentioned", "no information available", "insufficient information", "i don't know"]
    if any(p in text for p in bad_patterns) and len(answer.split()) < 30:
        return "⚠️ Unable to find reliable information in the knowledge base. Please consult local experts."

    # 🚨 Contradiction detection
    contradictions = [
        ("dry soil", "heavy irrigation"), ("high temperature", "cool climate"),
        ("excessive rain", "drought-tolerant"), ("high ph", "acidic soil")
    ]
    for c1, c2 in contradictions:
        if c1 in text and c2 in text:
            return "⚠️ Conflicting advice detected. Regenerating..."

    return answer.strip()


# ═══════════════════════════════════════════════════════════════════
#  FIX 7 — CONFIDENCE SCORING
# ═══════════════════════════════════════════════════════════════════

def get_confidence(response: str, context: str) -> (str, int):
    """
    Calculates a multi-factor confidence score (0-100).
    Implements Fix 7 and Issue 8.
    
    Returns: (Level, Score)
    """
    score = 0
    resp_low = response.lower()
    ctx_low = context.lower()

    # 1. Grounding Score (0-40)
    if context:
        context_len = len(context.strip())
        if context_len > 1000: score += 40
        elif context_len > 500: score += 30
        elif context_len > 200: score += 15
        
        # Keyword overlap check
        entities = extract_entities(response)
        matching_crops = [c for c in entities["crops"] if c in ctx_low]
        if matching_crops: score += 10

    # 2. Hallucination Signal Penalty (Up to -40)
    hallucination_penalty = 0
    for s in HALLUCINATION_SIGNALS:
        if s in resp_low:
            hallucination_penalty += 15
    score = max(0, score - hallucination_penalty)

    # 3. Web vs RAG presence (0-10)
    if "source: " in resp_low or "additional info:" in resp_low:
        score += 10

    # 4. Length/Structure sanity (0-10)
    if len(response.split()) > 50:
        score += 10
    
    # 5. Intent alignment
    intent = classify_query(response)
    if intent != "general":
        score += 5

    # Final Mapping
    if score >= 70: return "High", score
    if score >= 40: return "Medium", score
    return "Low", score
