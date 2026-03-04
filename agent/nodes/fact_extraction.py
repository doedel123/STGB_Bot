import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState, Fact, Allegation, Citation
from prompts.fact_extraction import SYSTEM_PROMPT
from services.gemini_client import llm_with_fallback as llm, extract_text

logger = logging.getLogger(__name__)

_LEGAL_KEYWORDS = {
    "vorsatz",
    "fahrlass",
    "arglist",
    "gewerbsmaessig",
    "gewerbsmassig",
    "planmaessig",
    "taeusch",
    "bereicherungsabsicht",
    "rechtswidrig",
    "schuld",
    "tatbestand",
    "strafbar",
}

_INFERENCE_PHRASES = {
    "ergibt sich",
    "offensichtlich",
    "musste wissen",
    "haette wissen koennen",
    "zielte darauf ab",
    "ist davon auszugehen",
    "spricht dafuer",
}


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_entities(text: str) -> dict[str, list[str]]:
    persons = sorted(
        {
            m.group(0)
            for m in re.finditer(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text)
        }
    )[:20]

    dates = sorted(
        {
            m.group(0)
            for m in re.finditer(
                r"\b(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})\b", text
            )
        }
    )[:30]

    places = sorted(
        {
            m.group(1)
            for m in re.finditer(r"\bin\s+([A-Z][\w-]+)", text)
        }
    )[:30]

    case_numbers = sorted(
        {
            m.group(0)
            for m in re.finditer(r"\b\d+\s?[A-Z]{1,5}\s?\d+/\d+\b", text)
        }
    )[:20]

    return {
        "persons": persons,
        "dates": dates,
        "places": places,
        "case_numbers": case_numbers,
    }


def _is_allegation(sentence: str) -> bool:
    lower = sentence.lower()
    if any(k in lower for k in _INFERENCE_PHRASES):
        return True
    if any(k in lower for k in _LEGAL_KEYWORDS):
        return True
    return bool(re.search(r"\bbehauptet\b|\bunterstellt\b|\bstaatsanwaltschaft\b", lower))


def _allegation_type(sentence: str) -> str:
    lower = sentence.lower()
    if any(k in lower for k in _INFERENCE_PHRASES):
        return "inference"
    if any(k in lower for k in _LEGAL_KEYWORDS):
        return "legal_conclusion"
    return "factual_claim"


def _heuristic_extract(raw_text: str) -> dict[str, Any]:
    normalized = _normalize_text(raw_text)
    candidates = [
        c.strip(" -\t")
        for c in re.split(r"(?<=[.!?])\s+|\n+", normalized)
        if c and len(c.strip()) >= 20
    ]

    facts: list[Fact] = []
    allegations: list[Allegation] = []

    for sentence in candidates[:250]:
        if _is_allegation(sentence):
            idx = len(allegations) + 1
            allegations.append(
                {
                    "id": f"A{idx}",
                    "text": sentence,
                    "type": _allegation_type(sentence),
                    "quote": sentence[:240],
                }
            )
        else:
            idx = len(facts) + 1
            facts.append(
                {
                    "id": f"F{idx}",
                    "text": sentence,
                    "quote": sentence[:240],
                }
            )

    if not facts and candidates:
        facts.append({"id": "F1", "text": candidates[0], "quote": candidates[0][:240]})

    entities = _extract_entities(normalized)
    return {
        "facts": facts,
        "allegations": allegations,
        "entities": entities,
    }


def _coerce_fact(item: dict[str, Any], index: int) -> Fact:
    return {
        "id": str(item.get("id") or f"F{index}"),
        "text": str(item.get("text") or "").strip(),
        "page": str(item.get("page") or "").strip() or None,
        "page_range": str(item.get("page_range") or "").strip() or None,
        "quote": str(item.get("quote") or "").strip() or None,
    }


def _coerce_allegation(item: dict[str, Any], index: int) -> Allegation:
    text = str(item.get("text") or "").strip()
    a_type = str(item.get("type") or "").strip().lower()
    if a_type not in {"legal_conclusion", "factual_claim", "inference"}:
        a_type = _allegation_type(text) if text else "factual_claim"

    return {
        "id": str(item.get("id") or f"A{index}"),
        "text": text,
        "type": a_type,
        "page": str(item.get("page") or "").strip() or None,
        "page_range": str(item.get("page_range") or "").strip() or None,
        "quote": str(item.get("quote") or "").strip() or None,
    }


def _build_citations(facts: list[Fact], allegations: list[Allegation]) -> list[Citation]:
    citations: list[Citation] = []

    for item in facts:
        quote = item.get("quote")
        if quote:
            citations.append(
                {
                    "id": item.get("id", ""),
                    "source_type": "document",
                    "source": "PDF",
                    "page": item.get("page") or item.get("page_range") or "",
                    "quote": quote,
                }
            )

    for item in allegations:
        quote = item.get("quote")
        if quote:
            citations.append(
                {
                    "id": item.get("id", ""),
                    "source_type": "document",
                    "source": "PDF",
                    "page": item.get("page") or item.get("page_range") or "",
                    "quote": quote,
                }
            )

    return citations[:80]


def fact_extraction_node(state: AgentState) -> dict:
    """Extract facts vs allegations from raw OCR text."""
    raw_text = state.get("raw_text") or state.get("pdf_content")
    if not raw_text:
        return {"error": "Keine Rohdaten fuer Faktenextraktion verfuegbar."}

    logger.info("fact_extraction: start")

    data: dict[str, Any] | None = None
    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Dokumenttext:\n\n{raw_text[:30000]}"),
            ]
        )
        data = _parse_json_payload(extract_text(response))
    except Exception as exc:
        logger.warning("fact_extraction: llm failed, falling back to heuristics (%s)", exc)

    if not data:
        data = _heuristic_extract(raw_text)

    facts_raw = data.get("facts") if isinstance(data.get("facts"), list) else []
    allegations_raw = data.get("allegations") if isinstance(data.get("allegations"), list) else []

    facts = [_coerce_fact(item, i) for i, item in enumerate(facts_raw, 1) if isinstance(item, dict)]
    allegations = [
        _coerce_allegation(item, i)
        for i, item in enumerate(allegations_raw, 1)
        if isinstance(item, dict)
    ]

    facts = [f for f in facts if f.get("text")]
    allegations = [a for a in allegations if a.get("text")]

    if not facts and not allegations:
        fallback = _heuristic_extract(raw_text)
        facts = fallback.get("facts", [])
        allegations = fallback.get("allegations", [])
        entities = fallback.get("entities", {})
    else:
        entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}

    citations = _build_citations(facts, allegations)

    logger.info(
        "fact_extraction: completed facts=%d allegations=%d citations=%d",
        len(facts),
        len(allegations),
        len(citations),
    )

    structure = dict(state.get("document_structure") or {})
    structure["entities"] = entities

    return {
        "facts": facts,
        "allegations": allegations,
        "citations": citations,
        "document_structure": structure,
    }
