"""
AgriSense-AI — crop_engine.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge.crop_info import CROP_MASTER

# Build CROP_DB dynamically from CROP_MASTER for a single source of truth across all regions
CROP_DB = []
for key, info in CROP_MASTER.items():
    CROP_DB.append({
        "name": info["canonical"],
        "region": ["all", "global", "india"],  # Unbiased global default
        "season": ["kharif", "rabi", "summer", "all"], 
        "ph_range": (info["ph"]["min"], info["ph"]["max"]),
        "water": "high" if info["rain"]["min"] > 1000 else "low",
        "type": info["type"].lower()
    })

def filter_crops(region, month, ph=None, water=None):
    results = []

    for crop in CROP_DB:
        # region match - relaxed to support all regions uniformly
        region_list = [r.lower() for r in crop["region"]]
        
        # We assume crops can grow if the region constraint is empty or matches global aliases
        if region and region.lower() not in region_list and "all" not in region_list:
            continue

        # season match
        season_list = [s.lower() for s in crop["season"]]
        if month and month.lower() not in season_list and "all" not in season_list:
            continue

        # pH match
        if ph is not None:
            lo, hi = crop["ph_range"]
            if not (lo <= ph <= hi):
                continue

        # water match optional
        if water and crop["water"] != water:
            continue

        results.append(crop)

    return results

def prioritize(crops):
    # UNBIASED PRIORITY: Sort by name or purely return as-is rather than favoring 'cash' crops
    return sorted(crops, key=lambda x: x.get("name", ""))

def validate_output(crops):
    if len(crops) < 3:
        return False
    return True

def add_water_context(crops):
    irrigated = [c["name"] for c in crops if c.get("water") == "high"]
    rainfed   = [c["name"] for c in crops if c.get("water") == "low"]
    return irrigated, rainfed
