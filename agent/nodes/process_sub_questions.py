"""Process all sub-questions in parallel.

For each sub-question the following steps run concurrently:
  1. RAG retrieval (RAGIE) + Web search (Gemini Google Search) — in parallel
  2. Synthesis (Gemini) — after both sources are available

All sub-questions are processed at the same time via ``asyncio.gather``,
which dramatically reduces total latency compared to the sequential loop.
"""

import asyncio

from langchain_core.messages import SystemMessage, HumanMessage

from agent.state import AgentState
from prompts.synthesize import SYSTEM_PROMPT
from services.gemini_client import llm, extract_text, search_with_grounding
from services.ragie_client import retrieve, format_chunks


# ── Per-sub-question helpers ─────────────────────────────────────────


async def _retrieve_rag(sub_q: dict) -> str:
    """Run RAG retrieval for a single sub-question (async wrapper)."""
    context_needed = sub_q.get("context_needed", "both")

    try:
        if context_needed == "both":
            # Query both partitions in parallel
            chunks_stgb, chunks_stpo = await asyncio.gather(
                asyncio.to_thread(retrieve, sub_q["question"], 4, "stgb"),
                asyncio.to_thread(retrieve, sub_q["question"], 4, "stpo"),
            )
            chunks = chunks_stgb + chunks_stpo
            chunks.sort(key=lambda c: c["score"], reverse=True)
        else:
            chunks = await asyncio.to_thread(
                retrieve, sub_q["question"], 6, context_needed
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


async def _synthesize(sub_q: dict, summary: str) -> str:
    """Create a partial analysis for a single sub-question (async)."""
    user_msg = (
        f"## Sachverhalt\n{summary}\n\n"
        f"## Rechtsfrage\n{sub_q['question']}\n\n"
        f"## Kommentarliteratur (StGB/StPO)\n{sub_q.get('rag_results', 'Nicht verfuegbar')}\n\n"
        f"## Aktuelle Rechtsprechung\n{sub_q.get('search_results', 'Nicht verfuegbar')}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    return extract_text(response)


async def _process_single(sub_q: dict, summary: str) -> dict:
    """Process one sub-question end-to-end.

    Step 1: RAG + Web Search in parallel
    Step 2: Synthesis (needs both results)
    """
    # Step 1 — retrieve commentary + case law concurrently
    rag_results, search_results = await asyncio.gather(
        _retrieve_rag(sub_q),
        _search_case_law(sub_q, summary),
    )

    updated = {**sub_q, "rag_results": rag_results, "search_results": search_results}

    # Step 2 — synthesize using both sources
    synthesis = await _synthesize(updated, summary)
    updated["synthesis"] = synthesis

    return updated


# ── LangGraph node ───────────────────────────────────────────────────


async def process_sub_questions_node(state: AgentState) -> dict:
    """Process ALL sub-questions in parallel and return the completed list."""
    summary = state.get("document_summary", "")
    sub_questions = state["sub_questions"]

    results = await asyncio.gather(
        *[_process_single(sq, summary) for sq in sub_questions]
    )

    return {
        "sub_questions": list(results),
        "current_sub_q_index": len(results),
    }
