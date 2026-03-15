"""Process all sub-questions in parallel.

For each sub-question the following steps run concurrently:
  1. RAG retrieval (RAGIE) + Web search (Gemini Google Search) — in parallel
  2. Synthesis (active provider) — after both sources are available

All sub-questions are processed at the same time via ``asyncio.gather``,
which dramatically reduces total latency compared to the sequential loop.
"""

import asyncio
import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from prompts.synthesize import SYSTEM_PROMPT
from services.gemini_client import (
    llm_with_fallback as llm,
    extract_text,
    search_with_grounding,
    ensure_provider,
)
from services.ragie_client import retrieve, format_chunks


# ── Per-sub-question helpers ─────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _match_items_for_question(
    question: str, items: list[dict], text_key: str = "text", limit: int = 6
) -> list[dict]:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return items[:limit]

    scored: list[tuple[float, dict]] = []
    for item in items:
        item_text = str(item.get(text_key, ""))
        item_tokens = _tokenize(item_text)
        if not item_tokens:
            continue
        overlap = len(q_tokens & item_tokens)
        if overlap == 0:
            continue
        score = overlap / len(q_tokens | item_tokens)
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


async def _retrieve_rag(sub_q: dict) -> str:
    """Run RAG retrieval for a single sub-question (async wrapper)."""
    try:
        chunks = await asyncio.to_thread(
            retrieve, sub_q["question"], 6, "strafrecht"
        )
        return format_chunks(chunks)
    except Exception as e:
        return f"RAG-Abfrage fehlgeschlagen: {e}"


async def _search_case_law(sub_q: dict, summary: str) -> str:
    """Run web search for a single sub-question (async wrapper)."""
    query = (
        f"Finde aktuelle deutsche Rechtsprechung und Urteile zu folgender "
        f"Rechtsfrage im deutschen Strafrecht:\n\n"
        f"{sub_q['question']}\n\n"
        f"Kontext: {summary}"
    )

    try:
        result = await asyncio.to_thread(search_with_grounding, query)
        search_text = result["text"]
        if result["sources"]:
            search_text += "\n\nQuellen:\n"
            for src in result["sources"]:
                search_text += f"- {src['title']}: {src['uri']}\n"
        return search_text
    except Exception as e:
        return f"Websuche fehlgeschlagen: {e}"


async def _synthesize(
    sub_q: dict,
    summary: str,
    facts: list[dict],
    allegations: list[dict],
    validation_report: dict,
    global_issues: list[str],
    provider: str,
) -> str:
    """Create a partial analysis for a single sub-question (async)."""
    ensure_provider(provider)
    related_facts = _match_items_for_question(sub_q["question"], facts)
    related_allegations = _match_items_for_question(sub_q["question"], allegations)
    related_validation = {
        a.get("id"): validation_report.get(a.get("id"))
        for a in related_allegations
        if a.get("id") in validation_report
    }

    issues_to_check = sub_q.get("issues_to_check") or global_issues

    user_msg = (
        f"## Sachverhalt\n{summary}\n\n"
        f"## Rechtsfrage\n{sub_q['question']}\n\n"
        f"## Relevante Fakten\n{json.dumps(related_facts, ensure_ascii=False)}\n\n"
        f"## Relevante Behauptungen\n{json.dumps(related_allegations, ensure_ascii=False)}\n\n"
        f"## Issues to check\n{json.dumps(issues_to_check, ensure_ascii=False)}\n\n"
        f"## Vorliegende Validierungshinweise\n{json.dumps(related_validation, ensure_ascii=False)}\n\n"
    )

    if provider == "openai":
        user_msg += (
            "## Recherche-Anweisung\n"
            "Nutze das file_search-Tool, um in der StGB/StPO-Kommentarliteratur "
            "nach relevanten Kommentierungen zu den betroffenen Normen zu suchen.\n"
            "Nutze das web_search-Tool, um aktuelle deutsche Rechtsprechung und "
            "Urteile zur Rechtsfrage zu finden.\n"
            "Zitiere alle gefundenen Quellen konkret."
        )
    else:
        user_msg += (
            f"## Kommentarliteratur (StGB/StPO)\n{sub_q.get('rag_results', 'Nicht verfuegbar')}\n\n"
            f"## Aktuelle Rechtsprechung\n{sub_q.get('search_results', 'Nicht verfuegbar')}"
        )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    return extract_text(response)


async def _process_single(
    sub_q: dict,
    summary: str,
    facts: list[dict],
    allegations: list[dict],
    validation_report: dict,
    global_issues: list[str],
    provider: str,
) -> dict:
    """Process one sub-question end-to-end.

    When provider is "openai", GPT uses built-in file_search + web_search tools
    directly, so manual RAGIE/Gemini retrieval is skipped.
    """
    if provider == "openai":
        # OpenAI handles retrieval via built-in tools during synthesis
        updated = {**sub_q, "rag_results": None, "search_results": None}
    else:
        # Gemini path: manual RAG + web search concurrently
        rag_results, search_results = await asyncio.gather(
            _retrieve_rag(sub_q),
            _search_case_law(sub_q, summary),
        )
        updated = {**sub_q, "rag_results": rag_results, "search_results": search_results}

    # Step 2 — synthesize using both sources
    synthesis = await _synthesize(
        updated,
        summary,
        facts,
        allegations,
        validation_report,
        global_issues,
        provider,
    )
    updated["synthesis"] = synthesis

    return updated


# ── LangGraph node ───────────────────────────────────────────────────


async def process_sub_questions_node(state: AgentState) -> dict:
    """Process ALL sub-questions in parallel and return the completed list."""
    provider = state.get("provider", "gemini")
    ensure_provider(provider)
    summary = state.get("document_summary", "")
    sub_questions = state["sub_questions"]
    facts = list(state.get("facts") or [])
    allegations = list(state.get("allegations") or [])
    validation_report = dict(state.get("validation_report") or {})
    global_issues = list(state.get("issues_to_check") or [])

    results = await asyncio.gather(
        *[
            _process_single(
                sq,
                summary,
                facts,
                allegations,
                validation_report,
                global_issues,
                provider,
            )
            for sq in sub_questions
        ]
    )

    return {
        "sub_questions": list(results),
        "current_sub_q_index": len(results),
    }
