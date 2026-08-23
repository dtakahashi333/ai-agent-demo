# tools/search_qdrant.py
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

model = SentenceTransformer("llm/BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("llm/BAAI/bge-reranker-base")

vecdb_client = QdrantClient("localhost", port=6333)


def search_qdrant(query: str) -> list[str]:
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = vecdb_client.query_points(
        collection_name="squad",
        query=query_vector,
        limit=20,
    ).points

    contexts = []

    # Rerank the contexts
    pairs = [(query, result.payload["text"]) for result in results]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)

    # retrieved_context = "\n\n".join(context for context, _ in ranked[:5])

    return [context for context, _ in ranked[:5]]
