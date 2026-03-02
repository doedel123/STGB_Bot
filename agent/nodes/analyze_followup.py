import json

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from prompts.followup import SYSTEM_PROMPT_FOLLOWUP_ANALYZE
from services.gemini_client import llm, extract_text


def analyze_followup_node(state: AgentState) -> dict:
    """Decompose a follow-up question into sub-questions using the existing document summary.

    This is a lighter version of ``analyze_document_node`` that does NOT send
    the full PDF text to the LLM — only the summary — making it faster and cheaper.
    """
    user_query = state["user_query"]
    document_summary = state.get("document_summary", "")
    previous_analysis = state.get("previous_analysis", "")

    system_prompt = SYSTEM_PROMPT_FOLLOWUP_ANALYZE.format(
        document_summary=document_summary,
    )

    user_msg = f"Nachfolgefrage: {user_query}"
    if previous_analysis:
        # Include a truncated excerpt of the prior analysis for context
        truncated = previous_analysis[:2000]
        user_msg += f"\n\nBisherige Analyse (Auszug):\n{truncated}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])

    text = extract_text(response)
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"error": f"Followup-Analyse konnte nicht geparst werden: {text[:200]}"}

    sub_questions = []
    for sq in data.get("sub_questions", []):
        sub_questions.append({
            "question": sq["question"],
            "context_needed": sq.get("context_needed", "both"),
            "rag_results": None,
            "search_results": None,
            "synthesis": None,
        })

    if not sub_questions:
        return {"error": "Keine Teilfragen fuer die Nachfolgefrage generiert."}

    return {
        "sub_questions": sub_questions,
        "current_sub_q_index": 0,
    }
