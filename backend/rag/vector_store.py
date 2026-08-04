"""
AgriSense-AI — vector_store.py
Pre-loads models at import time to prevent first-query latency.
"""

from pathlib import Path
import logging
from rag.embeddings import get_embedding_model
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

DB_PATH = str(Path(__file__).parent / "db")

# 🧠 Internal instances for caching
_embedding = None
_vectorstore = None

def get_embedding():
    """Lazily initialize or return the pre-loaded HuggingFace embedding model."""
    global _embedding
    if _embedding is None:
        logger.info("Initializing OpenAI Embedding Model (text-embedding-3-small)...")
        _embedding = get_embedding_model()
    return _embedding

def get_vectorstore():
    """Lazily initialize or return the pre-loaded Chroma vector store."""
    global _vectorstore
    if _vectorstore is None:
        logger.info(f"Connecting to Chroma DB at {DB_PATH}...")
        _vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=get_embedding()
        )
    return _vectorstore

def warmup():
    """Trigger initialization of all heavy components."""
    logger.info("🚀 Pre-warming AI components...")
    get_vectorstore()
    logger.info("✅ AI components ready for requests.")
