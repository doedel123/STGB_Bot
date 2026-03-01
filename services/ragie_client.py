from ragie import Ragie

from utils.config import RAGIE_API_KEY

_client = Ragie(auth=RAGIE_API_KEY)


def retrieve(query: str, top_k: int = 6) -> list[dict]:
    """Retrieve relevant chunks from RAGIE RAG.

    Returns a list of dicts with keys: text, score, document_name.
    """
    response = _client.retrievals.retrieve(
        request={
            "query": query,
            "rerank": True,
            "top_k": top_k,
            "max_chunks_per_document": 3,
        }
    )

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
