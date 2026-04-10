"""
AgriSense-AI — ingest.py
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path
from rag.embeddings import get_embedding_model

# Paths
DATA_PATH = Path(__file__).parent / "data" / "agriculture.pdf"
DB_PATH = Path(__file__).parent / "db"


def ingest():

    print("📄 Loading PDF...")
    loader = PyPDFLoader(str(DATA_PATH))
    docs = loader.load()

    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    print(f"🔢 Total chunks: {len(chunks)}")

    embedding = get_embedding_model()

    print("🧠 Creating vector DB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(DB_PATH)
    )

    vectordb.persist()

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    ingest()
