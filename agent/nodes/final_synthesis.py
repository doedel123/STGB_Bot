from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from prompts.final_analysis import SYSTEM_PROMPT
from services.gemini_client import llm_with_fallback as llm, extract_text


def final_synthesis_node(state: AgentState) -> dict:
    """Produce the comprehensive legal opinion from all sub-question syntheses."""
    summary = state.get("document_summary", "")

    parts = [f"## Sachverhalt\n{summary}\n"]
    for i, sq in enumerate(state["sub_questions"], 1):
        parts.append(
            f"## Teilanalyse {i}: {sq['question']}\n"
            f"{sq.get('synthesis', 'Nicht verfuegbar')}"
        )

    user_msg = "\n\n".join(parts)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    return {"final_analysis": extract_text(response)}
