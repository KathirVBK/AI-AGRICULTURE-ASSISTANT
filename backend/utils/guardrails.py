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
    "tractor", "machinery", "plough", "plow", "sowing", "sow", "organic",
    "hydroponics", "aquaponics", "poultry", "livestock", "dairy", "cattle",
    # Specific crops & plurals
    "rice", "wheat", "paddy", "maize", "corn", "cotton", "sugarcane",
    "banana", "bananas", "coconut", "coconuts", "sorghum", "millet", "millets", "turmeric",
    "groundnut", "soybean", "mustard", "sunflower", "potato", "potatoes", "tomato", "tomatoes",
    "onion", "onions", "chilies", "chilli", "chillies", "coffee", "tea", "rubber", "jute",
    "pigeon pea", "arhar", "chickpea", "lentil", "cowpea", "barley",
    "mango", "mangoes", "guava", "papaya", "citrus", "apple", "apples", "grape", "grapes", "strawberry",
    # Topics, pathology & care
    "nutrient", "nitrogen", "phosphorus", "potassium", "npk", "compost",
    "manure", "organic", "drip", "sprinkler", "weed", "weedicide",
    "fungicide", "disease", "diseases", "blight", "rot", "insect", "aphid", "thrips",
    "virus", "viruses", "leaf", "leaves", "curl", "wilt", "wilting", "treatment", "treatments",
    "symptom", "symptoms", "infection", "intercrop", "intercropping", "crop rotation", "mulch", "mulching",
    "water", "drought", "flood", "rainfed", "irrigated", "greenhouse",
    "germinate", "tilling", "vermicompost", "sericulture", "apiculture",
]

# Terms that explicitly signal a non-agri or off-topic query
NON_AGRI_SIGNALS = [
    "president", "prime minister", "movie", "film", "actor", "actress",
    "politics", "celebrity", "sports", "cricket", "football", "coding",
    "python programming", "javascript", "software", "hardware", "physics",
    "chemistry", "math", "history", "geography", "space shuttle",
    "rocket", "nasa", "planet", "galaxy", "bitcoin", "crypto", "stock market",
    "fashion", "gaming", "esports", "music", "song", "lyrics", "joke",
    "travel", "hotels", "flight", "tickets", "medicine", "doctor", "health",
    "recipe", "cooking", "chef", "restaurant", "hair", "skin", "muscle",
    "weight loss", "fitness", "exercise",
]

# Ambiguous words that look agri but aren't in this context
AMBIGUOUS = ["space", "outer space", "universe", "moon", "mars"]


# Conversational steering / follow-up keywords allowed when in chat
CONVERSATIONAL_STEERING = [
    "step", "steps", "guide", "guided", "guidance", "first", "second", "next",
    "continue", "how", "process", "beginner", "expert", "explain", "details",
    "tell me", "start", "proceed", "option", "yes", "ok", "okay", "done", "finished"
]


def is_agri_query(query: str, history: list = None) -> bool:
    """
    Determines if a query is related to agriculture or is a valid conversational steering instruction.
    Uses a combination of keyword matching, chat history context, and signal detection.
    """
    q = query.lower().strip()
    clean_q = re.sub(r'[^\w\s]', '', q)
    
    # 1. Allow common greetings
    if clean_q in ["hello", "hi", "hey", "help", "thanks", "thank you", "good morning", "good evening"]:
        return True

    # 2. Allow conversational steering/follow-ups if chat history exists or if query has steering words
    if history and len(history) > 0:
        # User is in an ongoing farming consultation session — allow follow-ups unless explicitly off-topic
        has_non_agri = any(re.search(rf"\b{kw}\b", q) for kw in NON_AGRI_SIGNALS)
        if not has_non_agri:
            return True

    # Allow steering phrases (e.g. "guide me step by step")
    has_steering = any(re.search(rf"\b{kw}\b", q) for kw in CONVERSATIONAL_STEERING)
    if has_steering:
        has_non_agri = any(re.search(rf"\b{kw}\b", q) for kw in NON_AGRI_SIGNALS)
        if not has_non_agri:
            return True

    # 3. Check space/planet terms
    if any(word in q for word in AMBIGUOUS):
        if not any(word in q for word in ["earth", "soil", "terrestrial", "land"]):
            return False

    has_agri = any(re.search(rf"\b{kw}\b", q) for kw in AGRI_KEYWORDS)
    has_non_agri = any(re.search(rf"\b{kw}\b", q) for kw in NON_AGRI_SIGNALS)
    
    # Rejection logic
    if has_non_agri:
        strong_agri = ["crop", "soil", "fertilizer", "irrigation", "pest", "pesticide", "harvest", "npk"]
        if not any(re.search(rf"\b{kw}\b", q) for kw in strong_agri):
            return False
        
    if not has_agri and not has_steering:
        return False
        
    return True


def guardrail_response(query: str, history: list = None):
    """
    Returns (allowed: bool, message: str | None).
    If allowed=False, message contains the professional refusal.
    """
    if not is_agri_query(query, history=history):
        return False, (
            "❌ I am the AgriSense AI Assistant, strictly dedicated to agricultural guidance.\n\n"
            "I can only help with agriculture-based questions, such as:\n"
            "  • Crop selection, sowing, and cultivation steps\n"
            "  • Soil health, pH levels, and fertilizer management (NPK)\n"
            "  • Pest, disease, and weed control strategies\n"
            "  • Irrigation, weather impacts, and harvesting advice\n\n"
            "Please rephrase your question to focus on these agricultural topics."
        )
    return True, None

