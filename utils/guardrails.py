"""
AgriSense-AI — guardrails.py
Strict domain guardrails to ensure only agricultural queries are processed.
"""

import re

AGRI_KEYWORDS = [
    # Core topics
    "crop", "crops", "soil", "fertilizer", "fertiliser", "irrigation",
    "pest", "pesticide", "farming", "agriculture", "harvest", "harvesting",
    "yield", "plant", "planting", "seed", "seeds", "weather", "rainfall",
    "farm", "vegetable", "vegetables", "fruit", "fruits", "ph",
    "humidity", "temperature", "cultivate", "cultivation", "grow", "growing",
    "season", "region", "climate", "field", "garden", "greenhouse",
    # Specific crops
    "rice", "wheat", "paddy", "maize", "corn", "cotton", "sugarcane",
    "banana", "coconut", "sorghum", "millet", "millets", "turmeric",
    "groundnut", "soybean", "mustard", "sunflower", "potato", "tomato",
    "onion", "chilies", "chilli", "coffee", "tea", "rubber", "jute",
    "pigeon pea", "arhar", "chickpea", "lentil", "cowpea", "barley",
    # Topics
    "nutrient", "nitrogen", "phosphorus", "potassium", "npk", "compost",
    "manure", "organic", "drip", "sprinkler", "weed", "weedicide",
    "fungicide", "disease", "blight", "rot", "insect", "aphid", "thrips",
    "intercrop", "intercropping", "crop rotation", "mulch", "mulching",
    "water", "drought", "flood", "rainfed", "irrigated", "greenhouse",
    "grow", "plant", "sow", "sowing", "germinate", "tilling", "plowing",
    "livestock", "poultry", "dairy", "manure", "vermicompost",
]

# Terms that explicitly signal a non-agri or off-topic query
NON_AGRI_SIGNALS = [
    "president", "prime minister", "movie", "film", "actor", "actress",
    "politics", "celebrity", "sports", "cricket", "football", "coding",
    "python programming", "javascript", "software", "hardware", "physics",
    "chemistry", "math", "history", "geography", "space shuttle",
    "rocket", "nasa", "planet", "galaxy", "bitcoin", "crypto", "stock market",
    "fashion", "gaming", "esports", "music", "song", "lyrics",
]

# Ambiguous words that look agri but aren't in this context
AMBIGUOUS = ["space", "outer space", "universe", "moon", "mars"]


def is_agri_query(query: str) -> bool:
    """
    Determines if a query is related to agriculture.
    Uses a combination of keyword matching and signal detection.
    """
    q = query.lower().strip()
    
    # Check for ambiguous space/planet terms
    if any(word in q for word in AMBIGUOUS):
        # Unless it specifically mentions "farming" or "crops" in a way that isn't space-related
        if not any(word in q for word in ["earth", "soil", "terrestrial"]):
            return False

    has_agri = any(re.search(rf"\b{kw}\b", q) for kw in AGRI_KEYWORDS)
    has_non_agri = any(re.search(rf"\b{kw}\b", q) for kw in NON_AGRI_SIGNALS)
    
    # If it has non-agri signals and no strong agri keywords, reject.
    if has_non_agri and not has_agri:
        return False
        
    # If it has no agri keywords at all, reject.
    if not has_agri:
        # Check if it's a greeting or very short common phrase
        if q in ["hello", "hi", "hey", "help", "thanks", "thank you"]:
            return True
        return False
        
    return True


def guardrail_response(query: str):
    """
    Returns (allowed: bool, message: str | None).
    If allowed=False, message contains the professional refusal.
    """
    if not is_agri_query(query):
        return False, (
            "❌ I am the AgriSense AI Assistant, strictly dedicated to agricultural guidance.\n\n"
            "I can assist with:\n"
            "  • Crop selection and cultivation steps\n"
            "  • Soil health and fertilizer management (NPK, pH)\n"
            "  • Pest, disease, and weed control\n"
            "  • Irrigation and weather-related advice\n\n"
            "Please rephrase your question to focus on these agricultural topics."
        )
    return True, None
