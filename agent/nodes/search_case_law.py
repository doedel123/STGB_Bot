from agent.state import AgentState
from services.gemini_client import search_with_grounding


def search_case_law_node(state: AgentState) -> dict:
    """Search for current case law using Gemini with Google Search grounding."""
    idx = state["current_sub_q_index"]
    sub_q = state["sub_questions"][idx]
    summary = state.get("document_summary", "")

    query = (
        f"Finde aktuelle deutsche Rechtsprechung und Urteile zu folgender "
        f"Rechtsfrage im deutschen Strafrecht:\n\n"
        f"{sub_q['question']}\n\n"
        f"Kontext: {summary}"
    )

    try:
        result = search_with_grounding(query)
        search_text = result["text"]
        if result["sources"]:
            search_text += "\n\nQuellen:\n"
            for src in result["sources"]:
                search_text += f"- {src['title']}: {src['uri']}\n"
    except Exception as e:
        search_text = f"Websuche fehlgeschlagen: {e}"

    sub_questions = list(state["sub_questions"])
    sub_questions[idx] = {**sub_questions[idx], "search_results": search_text}

    return {"sub_questions": sub_questions}
