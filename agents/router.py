"""
AgriSense-AI — router.py
"""

from core.pipeline import run_query

def route(query: str, history: list = None):
    """
    Deprecated universal router. 
    Now delegates directly to the strict Rules Engine pipeline.
    """
    return run_query(query)

