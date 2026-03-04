import logging
import re
from typing import Iterable

from agent.state import AgentState, Allegation, Contradiction, Fact, ValidationEntry, Citation

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "der",
    "die",
    "das",
    "und",
    "oder",
    "den",
    "dem",
    "ein",
    "eine",
    "einer",
    "eines",
    "einem",
    "mit",
    "von",
    "im",
    "in",
    "am",
    "an",
    "auf",
    "zu",
    "ist",
    "war",
    "wurde",
    "dass",
    "als",
    "nach",
    "vor",
    "fuer",
    "fur",
    "durch",
    "aus",
    "sich",
    "nicht",
    "kein",
    "keine",
    "einen",
    "bei",
}

_NEGATION_MARKERS = (" nicht ", " kein ", " keine ", " niemals ", " nie ", " ohne ")


def _tokenize(text: str) -> set[str]:
    tokens = {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t not in _STOPWORDS}
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _support_strength(score: float, matched: int) -> str:
    if matched <= 0:
        return "none"
    if score >= 0.42:
        return "strong"
    if score >= 0.26:
        return "medium"
    if score >= 0.14:
        return "weak"
    return "none"


def _extract_dates(text: str) -> set[str]:
    return {
        m.group(0)
        for m in re.finditer(r"\b(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})\b", text)
    }


def _polarity(text: str) -> int:
    normalized = f" {text.lower()} "
    if any(marker in normalized for marker in _NEGATION_MARKERS):
        return -1
    return 1


def _quote(item: dict) -> str:
    return str(item.get("quote") or item.get("text") or "")[:260]


def _iter_statements(facts: list[Fact], allegations: list[Allegation]) -> Iterable[dict]:
    for f in facts:
        yield {"kind": "fact", "id": f.get("id", ""), "text": f.get("text", ""), "quote": _quote(f)}
    for a in allegations:
        yield {
            "kind": "allegation",
            "id": a.get("id", ""),
            "text": a.get("text", ""),
            "quote": _quote(a),
        }


def _detect_contradictions(facts: list[Fact], allegations: list[Allegation]) -> list[Contradiction]:
    statements = list(_iter_statements(facts, allegations))
    contradictions: list[Contradiction] = []
    seen_keys: set[tuple[str, str]] = set()

    for i, left in enumerate(statements):
        left_text = left["text"]
        if not left_text:
            continue
        left_tokens = _tokenize(left_text)
        left_dates = _extract_dates(left_text)
        left_pol = _polarity(left_text)

        for right in statements[i + 1 :]:
            right_text = right["text"]
            if not right_text:
                continue
            right_tokens = _tokenize(right_text)
            overlap = left_tokens & right_tokens
            if len(overlap) < 2:
                continue

            right_dates = _extract_dates(right_text)
            right_pol = _polarity(right_text)
            key = tuple(sorted((left.get("id", ""), right.get("id", ""))))
            if key in seen_keys:
                continue

            contradiction: Contradiction | None = None

            if left_pol != right_pol:
                contradiction = {
                    "description": "Moeglicher Widerspruch: derselbe Geschehenskern wird einmal bejaht und einmal verneint.",
                }
            elif left_dates and right_dates and left_dates != right_dates:
                contradiction = {
                    "description": "Moeglicher Zeitwiderspruch: gleiche Handlung wird unterschiedlichen Daten zugeordnet.",
                }
            else:
                l = left_text.lower()
                r = right_text.lower()
                if ("wusste" in l and "wissen" in r and "koenn" in r) or (
                    "wusste" in r and "wissen" in l and "koenn" in l
                ):
                    contradiction = {
                        "description": "Wissenswiderspruch: sichere Kenntnis vs nur hypothetische Kenntnismoeglichkeit.",
                    }

            if not contradiction:
                continue

            seen_keys.add(key)
            contradiction_id = f"C{len(contradictions) + 1}"
            contradiction.update(
                {
                    "id": contradiction_id,
                    "involved_fact_ids": [
                        x
                        for x, kind in ((left.get("id", ""), left["kind"]), (right.get("id", ""), right["kind"]))
                        if kind == "fact" and x
                    ],
                    "involved_allegation_ids": [
                        x
                        for x, kind in ((left.get("id", ""), left["kind"]), (right.get("id", ""), right["kind"]))
                        if kind == "allegation" and x
                    ],
                    "evidence_quotes": [left["quote"], right["quote"]],
                }
            )
            contradictions.append(contradiction)

    return contradictions[:30]


def _validate_allegation(allegation: Allegation, facts: list[Fact]) -> ValidationEntry:
    allegation_text = allegation.get("text", "")
    allegation_tokens = _tokenize(allegation_text)

    scored: list[tuple[float, str]] = []
    for fact in facts:
        fact_text = fact.get("text", "")
        score = _jaccard(allegation_tokens, _tokenize(fact_text))
        if score > 0:
            scored.append((score, fact.get("id", "")))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:4]
    top_ids = [f_id for _, f_id in top if f_id]

    best_score = top[0][0] if top else 0.0
    strength = _support_strength(best_score, len(top_ids))

    lower = allegation_text.lower()
    circular = (
        ("ergibt sich" in lower or "offensichtlich" in lower or "musste wissen" in lower)
        and strength in {"weak", "none"}
    )

    if strength == "none":
        notes = "Unbelegt: keine hinreichend passenden Tatsachen gefunden."
    elif strength == "weak":
        notes = "Schwach belegt: nur lose Anschlussfakten, Kernelement bleibt offen."
    elif strength == "medium":
        notes = "Teilweise belegt, aber wesentliche Elemente sollten weiter nachgewiesen werden."
    else:
        notes = "Gut belegt durch mehrere konkrete Tatsachenangaben."

    if circular:
        notes += " Moeglicher Zirkelschluss/Spekulation in der Begruendung."

    return {
        "supporting_fact_ids": top_ids,
        "support_strength": strength,
        "notes": notes,
        "circular_reasoning": circular,
    }


def _build_contradiction_citations(contradictions: list[Contradiction]) -> list[Citation]:
    citations: list[Citation] = []
    for contradiction in contradictions:
        quotes = contradiction.get("evidence_quotes") or []
        for idx, quote in enumerate(quotes, 1):
            if quote:
                citations.append(
                    {
                        "id": f"{contradiction.get('id', 'C?')}-{idx}",
                        "source_type": "analysis",
                        "source": "contradiction_scan",
                        "quote": quote,
                        "note": contradiction.get("description", ""),
                    }
                )
    return citations[:60]


def allegation_validation_node(state: AgentState) -> dict:
    """Validate each allegation against extracted facts and detect contradictions."""
    facts = list(state.get("facts") or [])
    allegations = list(state.get("allegations") or [])

    logger.info("allegation_validation: start facts=%d allegations=%d", len(facts), len(allegations))

    validation_report: dict[str, ValidationEntry] = {}
    for allegation in allegations:
        allegation_id = allegation.get("id")
        if not allegation_id:
            continue
        validation_report[allegation_id] = _validate_allegation(allegation, facts)

    contradictions = _detect_contradictions(facts, allegations)

    existing_citations = list(state.get("citations") or [])
    contradiction_citations = _build_contradiction_citations(contradictions)

    logger.info(
        "allegation_validation: completed validation=%d contradictions=%d",
        len(validation_report),
        len(contradictions),
    )

    return {
        "validation_report": validation_report,
        "contradictions": contradictions,
        "citations": (existing_citations + contradiction_citations)[:120],
    }
