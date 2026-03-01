from agent.state import AgentState
from services.ragie_client import retrieve, format_chunks


def retrieve_rag_node(state: AgentState) -> dict:
    """Query RAGIE for StGB/StPO commentary relevant to the current sub-question."""
    idx = state["current_sub_q_index"]
    sub_q = state["sub_questions"][idx]

    try:
        chunks = retrieve(sub_q["question"], top_k=6)
        context = format_chunks(chunks)
    except Exception as e:
        context = f"RAG-Abfrage fehlgeschlagen: {e}"

    sub_questions = list(state["sub_questions"])
    sub_questions[idx] = {**sub_questions[idx], "rag_results": context}

    return {"sub_questions": sub_questions}
