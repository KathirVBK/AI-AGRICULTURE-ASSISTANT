"""
AgriSense-AI — embeddings.py
"""

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

def get_embedding_model():
    load_dotenv()
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("NAVIGATE_API_KEY"),
        openai_api_base=os.getenv("NAVIGATE_BASE_URL")
    )
