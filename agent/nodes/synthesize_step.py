from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from prompts.synthesize import SYSTEM_PROMPT
from services.gemini_client import llm_with_fallback as llm, extract_text


def synthesize_step_node(state: AgentState) -> dict:
    """Synthesize RAG context + search results into a partial analysis for the current sub-question."""
    idx = state["current_sub_q_index"]
    sub_q = state["sub_questions"][idx]
    summary = state.get("document_summary", "")

    user_msg = (
        f"## Sachverhalt\n{summary}\n\n"
        f"## Rechtsfrage\n{sub_q['question']}\n\n"
        f"## Kommentarliteratur (StGB/StPO)\n{sub_q.get('rag_results', 'Nicht verfuegbar')}\n\n"
        f"## Aktuelle Rechtsprechung\n{sub_q.get('search_results', 'Nicht verfuegbar')}"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    sub_questions = list(state["sub_questions"])
    sub_questions[idx] = {**sub_questions[idx], "synthesis": extract_text(response)}

    return {
        "sub_questions": sub_questions,
        "current_sub_q_index": idx + 1,
    }
