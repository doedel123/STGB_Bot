import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from prompts.analyze_document import SYSTEM_PROMPT_FULL, SYSTEM_PROMPT_FOCUSED
from services.gemini_client import llm_with_fallback as llm, extract_text, ensure_provider

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def _compact_items(items: list[dict], title: str, limit: int = 10) -> str:
    if not items:
        return f"{title}: Keine"
    lines = [f"{title}:"]
    for item in items[:limit]:
        item_id = item.get("id", "?")
        text = str(item.get("text", "")).strip()
        if text:
            lines.append(f"- {item_id}: {text[:240]}")
    return "\n".join(lines)


def analyze_document_node(state: AgentState) -> dict:
    """Analyze extracted PDF text: summarize and decompose into sub-questions."""
    ensure_provider(state.get("provider"))
    pdf_content = state["pdf_content"]
    user_query = state.get("user_query")
    facts = list(state.get("facts") or [])
    allegations = list(state.get("allegations") or [])

    logger.info(
        "analyze_document: start user_query=%s facts=%d allegations=%d",
        bool(user_query),
        len(facts),
        len(allegations),
    )

    facts_summary = _compact_items(facts, "Extrahierte Fakten")
    allegations_summary = _compact_items(allegations, "Extrahierte Behauptungen")
    pre_context = f"{facts_summary}\n\n{allegations_summary}\n"

    # Choose prompt based on whether user asked a specific question
    if user_query:
        system_prompt = SYSTEM_PROMPT_FOCUSED
        user_msg = (
            f"Nutzerfrage: {user_query}\n\n"
            f"{pre_context}\n\n"
            f"Dokument:\n\n{pdf_content}"
        )
    else:
        system_prompt = SYSTEM_PROMPT_FULL
        user_msg = (
            f"{pre_context}\n\n"
            f"Analysiere folgendes Dokument:\n\n{pdf_content}"
        )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])

    # Parse the structured JSON response
    text = _strip_code_fences(extract_text(response))

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
            "issues_to_check": sq.get("issues_to_check") or [],
        })

    if not sub_questions:
        return {"error": "Keine Teilfragen generiert."}

    summary = data.get("summary", "")
    paragraphs = data.get("relevant_paragraphs", [])
    doc_type = data.get("document_type", "")
    accused = data.get("accused", [])
    issues_to_check = data.get("issues_to_check", [])

    full_summary = (
        f"Dokumenttyp: {doc_type}\n"
        f"Angeklagte: {', '.join(accused)}\n"
        f"Relevante Paragraphen: {', '.join(paragraphs)}\n\n"
        f"{summary}"
    )

    logger.info(
        "analyze_document: completed sub_questions=%d issues_to_check=%d",
        len(sub_questions),
        len(issues_to_check) if isinstance(issues_to_check, list) else 0,
    )

    return {
        "document_summary": full_summary,
        "sub_questions": sub_questions,
        "current_sub_q_index": 0,
        "issues_to_check": issues_to_check if isinstance(issues_to_check, list) else [],
    }
