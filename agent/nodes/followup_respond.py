from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.state import AgentState
from prompts.followup import SYSTEM_PROMPT_FOLLOWUP_RESPOND
from services.gemini_client import llm_with_fallback as llm, extract_text


def followup_respond_node(state: AgentState) -> dict:
    """Produce a focused answer to the follow-up question from sub-question syntheses.

    Unlike ``final_synthesis_node`` this does NOT generate a full Gutachten
    structure — it creates a concise, targeted response in Gutachtenstil that
    directly answers the user's follow-up question.
    """
    user_query = state["user_query"]
    summary = state.get("document_summary", "")

    parts = [
        f"## Sachverhalt\n{summary}\n",
        f"## Nachfolgefrage\n{user_query}\n",
    ]
    for i, sq in enumerate(state["sub_questions"], 1):
        parts.append(
            f"## Teilanalyse {i}: {sq['question']}\n"
            f"{sq.get('synthesis', 'Nicht verfuegbar')}"
        )

    user_msg = "\n\n".join(parts)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT_FOLLOWUP_RESPOND),
        HumanMessage(content=user_msg),
    ])

    content = extract_text(response)

    return {
        "final_analysis": content,
        "messages": [AIMessage(content=content)],
    }
