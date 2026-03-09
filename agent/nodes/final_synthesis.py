import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from prompts.final_analysis import SYSTEM_PROMPT
from services.gemini_client import llm_with_fallback as llm, extract_text, ensure_provider

logger = logging.getLogger(__name__)


def _summarize_validation(validation_report: dict) -> str:
    if not validation_report:
        return "Keine Validierungsdaten verfuegbar."
    return json.dumps(validation_report, ensure_ascii=False, indent=2)[:12000]


def final_synthesis_node(state: AgentState) -> dict:
    """Produce the comprehensive legal opinion from all sub-question syntheses."""
    ensure_provider(state.get("provider"))
    summary = state.get("document_summary", "")
    facts = state.get("facts", [])
    allegations = state.get("allegations", [])
    contradictions = state.get("contradictions", [])
    validation_report = state.get("validation_report", {})
    issues_to_check = state.get("issues_to_check", [])

    logger.info(
        "final_synthesis: start sub_questions=%d facts=%d allegations=%d contradictions=%d",
        len(state.get("sub_questions", [])),
        len(facts),
        len(allegations),
        len(contradictions),
    )

    parts = [
        f"## Sachverhalt\n{summary}\n",
        f"## Extrahierte Fakten\n{json.dumps(facts, ensure_ascii=False)}\n",
        f"## Extrahierte Behauptungen\n{json.dumps(allegations, ensure_ascii=False)}\n",
        f"## Globale Issues to check\n{json.dumps(issues_to_check, ensure_ascii=False)}\n",
        f"## Validation Report\n{_summarize_validation(validation_report)}\n",
        f"## Contradictions\n{json.dumps(contradictions, ensure_ascii=False)}\n",
    ]
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

    final_analysis = extract_text(response)
    logger.info("final_synthesis: completed final_length=%d", len(final_analysis))
    return {"final_analysis": final_analysis}
