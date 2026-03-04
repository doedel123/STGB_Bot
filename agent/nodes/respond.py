from langchain_core.messages import AIMessage

from agent.state import AgentState

DISCLAIMER = (
    "Keine Rechtsberatung. Bitte durch eine zugelassene Rechtsanwaeltin/"
    "einen zugelassenen Rechtsanwalt verifizieren lassen."
)


def _format_fehlerbericht(state: AgentState) -> str:
    validation = state.get("validation_report") or {}
    contradictions = state.get("contradictions") or []

    lines: list[str] = ["## Fehler-/Widerspruchsbericht"]
    lines.append("### Unbelegte Behauptungen")
    has_unbelegt = False
    for allegation_id, entry in validation.items():
        strength = str(entry.get("support_strength") or "none")
        if strength in {"none", "weak"}:
            has_unbelegt = True
            lines.append(
                f"- {allegation_id}: {strength} belegt | "
                f"Fakten: {entry.get('supporting_fact_ids', [])} | "
                f"Hinweis: {entry.get('notes', '')}"
            )
    if not has_unbelegt:
        lines.append("- Keine klar unbelegten Behauptungen erkannt.")

    lines.append("### Innere Widersprueche")
    if contradictions:
        for contradiction in contradictions:
            lines.append(
                f"- {contradiction.get('id', 'C?')}: {contradiction.get('description', '')} | "
                f"Belege: {contradiction.get('evidence_quotes', [])}"
            )
    else:
        lines.append("- Keine klaren inneren Widersprueche erkannt.")

    lines.append("### Zirkelschluesse / Spekulationen")
    has_circular = False
    for allegation_id, entry in validation.items():
        if entry.get("circular_reasoning"):
            has_circular = True
            lines.append(f"- {allegation_id}: {entry.get('notes', '')}")
    if not has_circular:
        lines.append("- Keine klaren Zirkelschluesse erkannt.")

    return "\n".join(lines)


def _format_citations(state: AgentState) -> str:
    citations = list(state.get("citations") or [])
    if not citations:
        return "## Citations\n- Keine strukturierten Zitate verfuegbar."

    lines = ["## Citations"]
    for citation in citations[:20]:
        cid = citation.get("id", "")
        page = citation.get("page", "")
        source = citation.get("source", citation.get("source_type", "quelle"))
        quote = citation.get("quote", "")
        note = citation.get("note", "")
        parts = [f"- {cid}" if cid else "-"]
        parts.append(f"[{source}]")
        if page:
            parts.append(f"({page})")
        if quote:
            parts.append(f"\"{quote}\"")
        if note:
            parts.append(f"- {note}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def respond_node(state: AgentState) -> dict:
    """Format the final analysis (or error) as a chat message."""
    if state.get("error"):
        content = f"Fehler bei der Analyse: {state['error']}"
    else:
        content = state.get("final_analysis", "Keine Analyse verfuegbar.")

    # Safety: ensure content is a plain string
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        content = "".join(parts)

    if not state.get("error"):
        if "Fehler-/Widerspruchsbericht" not in content:
            content = f"{content.rstrip()}\n\n{_format_fehlerbericht(state)}"
        if "## Citations" not in content and "## Quellen" not in content:
            content = f"{content.rstrip()}\n\n{_format_citations(state)}"
        if DISCLAIMER not in content:
            content = f"{content.rstrip()}\n\n{DISCLAIMER}"

    return {"messages": [AIMessage(content=content)]}
