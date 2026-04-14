"""
AgriSense-AI — crop_info.py
"""

# AgriSense-AI — knowledge/crop_info.py
# Centralized Ground Truth for Indian Crops

CROP_MASTER = {
    "rice": {
        "canonical": "Rice (Paddy)",
        "aliases": ["rice", "paddy", "nellu"],
        "temp": {"min": 20, "max": 35},
        "ph": {"min": 5.0, "max": 7.5},
        "rain": {"min": 1000, "max": 2500},
        "importance": 10,  # 1-10 scale
        "type": "Staple"
    },
    "sugarcane": {
        "canonical": "Sugarcane",
        "aliases": ["sugarcane", "karumbu"],
        "temp": {"min": 20, "max": 40},
        "ph": {"min": 6.5, "max": 7.5},
        "rain": {"min": 750, "max": 1200},
        "importance": 9,
        "type": "Cash Crop"
    },
    "turmeric": {
        "canonical": "Turmeric",
        "aliases": ["turmeric", "manjal"],
        "temp": {"min": 20, "max": 30},
        "ph": {"min": 4.5, "max": 7.5},
        "rain": {"min": 1500, "max": 2500},
        "importance": 8,
        "type": "Cash Crop"
    },
    "pigeon_pea": {
        "canonical": "Pigeon Pea (Arhar/Tur)",
        "aliases": ["pigeon pea", "tur", "arhar", "thandai"],
        "temp": {"min": 18, "max": 35},
        "ph": {"min": 5.0, "max": 8.0},
        "rain": {"min": 600, "max": 1000},
        "importance": 7,
        "type": "Pulse"
    },
    "sorghum": {
        "canonical": "Sorghum (Jowar)",
        "aliases": ["sorghum", "jowar", "cholam"],
        "temp": {"min": 25, "max": 35},
        "ph": {"min": 5.5, "max": 8.5},
        "rain": {"min": 400, "max": 600},
        "importance": 6,
        "type": "Millet"
    },
    "banana": {
        "canonical": "Banana",
        "aliases": ["banana", "vazhai"],
        "temp": {"min": 15, "max": 35},
        "ph": {"min": 6.5, "max": 7.5},
        "rain": {"min": 1500, "max": 2500},
        "importance": 9,
        "type": "Horticulture"
    },
    "coconut": {
        "canonical": "Coconut",
        "aliases": ["coconut", "thennai"],
        "temp": {"min": 20, "max": 32},
        "ph": {"min": 5.0, "max": 8.0},
        "rain": {"min": 1000, "max": 2500},
        "importance": 9,
        "type": "Plantation"
    },
    "maize": {
        "canonical": "Maize (Corn)",
        "aliases": ["maize", "corn", "makkacholam"],
        "temp": {"min": 18, "max": 35},
        "ph": {"min": 5.5, "max": 7.5},
        "rain": {"min": 600, "max": 1000},
        "importance": 7,
        "type": "Cereal"
    },
    "cotton": {
        "canonical": "Cotton",
        "aliases": ["cotton", "paruthi"],
        "temp": {"min": 21, "max": 30},
        "ph": {"min": 5.5, "max": 8.5},
        "rain": {"min": 500, "max": 1000},
        "importance": 8,
        "type": "Fiber"
    }
}

def get_crop_by_alias(alias):
    alias_low = alias.lower().strip()
    for key, info in CROP_MASTER.items():
        if alias_low in info["aliases"]:
            return info
    return None

def get_solution_space_diversity():
    """Returns a list of all canonical names grouped by importance and type."""
    data = []
    for k, v in CROP_MASTER.items():
        data.append({
            "name": v["canonical"],
            "type": v["type"],
            "importance": v["importance"],
            "requirements": {
                "temp": v["temp"],
                "ph": v["ph"],
                "rain": v["rain"]
            }
        })
    return data
