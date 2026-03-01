import json

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from prompts.analyze_document import SYSTEM_PROMPT_FULL, SYSTEM_PROMPT_FOCUSED
from services.gemini_client import llm, extract_text


def analyze_document_node(state: AgentState) -> dict:
    """Analyze extracted PDF text: summarize and decompose into sub-questions."""
    pdf_content = state["pdf_content"]
    user_query = state.get("user_query")

    # Choose prompt based on whether user asked a specific question
    if user_query:
        system_prompt = SYSTEM_PROMPT_FOCUSED
        user_msg = (
            f"Nutzerfrage: {user_query}\n\n"
            f"Dokument:\n\n{pdf_content}"
        )
    else:
        system_prompt = SYSTEM_PROMPT_FULL
        user_msg = f"Analysiere folgendes Dokument:\n\n{pdf_content}"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])

    # Parse the structured JSON response
    text = extract_text(response)
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"error": f"Dokumentanalyse konnte nicht geparst werden: {text[:200]}"}

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
        return {"error": "Keine Teilfragen generiert."}

    summary = data.get("summary", "")
    paragraphs = data.get("relevant_paragraphs", [])
    doc_type = data.get("document_type", "")
    accused = data.get("accused", [])

    full_summary = (
        f"Dokumenttyp: {doc_type}\n"
        f"Angeklagte: {', '.join(accused)}\n"
        f"Relevante Paragraphen: {', '.join(paragraphs)}\n\n"
        f"{summary}"
    )

    return {
        "document_summary": full_summary,
        "sub_questions": sub_questions,
        "current_sub_q_index": 0,
    }
