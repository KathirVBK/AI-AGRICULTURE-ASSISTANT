"""
AgriSense-AI — rag_pipeline.py
Semantic search with diversity-focused retrieval.
"""

from rag.vector_store import get_vectorstore


def retrieve(query: str, k=5):
    """
    Semantic search using MMR (Maximal Marginal Relevance) for diversity.
    """

    try:
        vectorstore = get_vectorstore()
        
        # Use MMR to ensure the retrieved documents are relevant but also diverse
        # to avoid context redundancy (e.g., getting 5 docs from the same page).
        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k*4,  # Fetch more docs and then pick the best diverse set
            lambda_mult=0.5  # 0.5 is a good balance between relevance and diversity
        )

        if not docs:
            return "", []

        # Deduplicate and clean text
        texts = []
        seen = set()
        for doc in docs:
            content = doc.page_content.strip()
            if content and content not in seen:
                texts.append(content)
                seen.add(content)

        context = "\n\n".join(texts)

        return context, texts

    except Exception as e:
        # Fallback to simple similarity search if MMR fails for any reason
        try:
            vectorstore = get_vectorstore()
            docs = vectorstore.similarity_search(query, k=k)
            texts = [doc.page_content for doc in docs]
            return "\n\n".join(texts), texts
        except Exception:
            raise RuntimeError(f"RAG retrieval failed: {str(e)}")
