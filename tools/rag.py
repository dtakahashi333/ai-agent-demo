# tools/rag.py
from .search_qdrant import search_qdrant


def search_documents(query: str) -> str:
    """
    Search the knowledge base and return relevant information.
    """

    results = search_qdrant(query)

    return "\n\n".join(result for result in results)
