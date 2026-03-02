from agent.state import AgentState
from services.ragie_client import retrieve, format_chunks


def retrieve_rag_node(state: AgentState) -> dict:
    """Query RAGIE for StGB/StPO commentary relevant to the current sub-question.

    Uses the sub-question's ``context_needed`` field to scope the search to
    the correct partition(s):
    - "stgb"  → only the stgb partition
    - "stpo"  → only the stpo partition
    - "both"  → queries both partitions and merges the results
    """
    idx = state["current_sub_q_index"]
    sub_q = state["sub_questions"][idx]
    context_needed = sub_q.get("context_needed", "both")

    try:
        if context_needed == "both":
            # Query both partitions and merge results
            chunks_stgb = retrieve(sub_q["question"], top_k=4, partition="stgb")
            chunks_stpo = retrieve(sub_q["question"], top_k=4, partition="stpo")
            chunks = chunks_stgb + chunks_stpo
            # Sort merged results by score (highest first)
            chunks.sort(key=lambda c: c["score"], reverse=True)
        else:
            chunks = retrieve(sub_q["question"], top_k=6, partition=context_needed)

        context = format_chunks(chunks)
    except Exception as e:
        context = f"RAG-Abfrage fehlgeschlagen: {e}"

    sub_questions = list(state["sub_questions"])
    sub_questions[idx] = {**sub_questions[idx], "rag_results": context}

    return {"sub_questions": sub_questions}
