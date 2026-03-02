from ragie import Ragie

from utils.config import RAGIE_API_KEY

_client = Ragie(auth=RAGIE_API_KEY)


def retrieve(query: str, top_k: int = 6, partition: str | None = None) -> list[dict]:
    """Retrieve relevant chunks from RAGIE RAG.

    Args:
        query: The search query.
        top_k: Maximum number of chunks to return.
        partition: Optional partition name to scope the search (e.g. "stgb", "stpo").

    Returns a list of dicts with keys: text, score, document_name.
    """
    request = {
        "query": query,
        "rerank": True,
        "top_k": top_k,
        "max_chunks_per_document": 3,
    }
    if partition:
        request["partition"] = partition

    response = _client.retrievals.retrieve(request=request)

    return [
        {
            "text": chunk.text,
            "score": chunk.score,
            "document_name": chunk.document_name,
        }
        for chunk in response.scored_chunks
    ]


def format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not chunks:
        return "Keine relevanten Kommentarstellen gefunden."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Quelle {i}: {chunk['document_name']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)
