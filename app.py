from engineio.payload import Payload
from pathlib import Path

Payload.max_decode_packets = 100  # default 16 is too low for parallel processing

import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.graph import graph, followup_graph

# Chainlit expects this parent dir to exist for per-session upload folders.
Path(__file__).resolve().parent.joinpath(".files").mkdir(parents=True, exist_ok=True)

# Display names for step indicators
_NODE_LABELS = {
    "extract_pdf": "PDF-Text wird extrahiert (Mistral OCR)...",
    "fact_extraction": "Fakten und Behauptungen werden extrahiert...",
    "analyze_document": "Dokument wird analysiert und Teilfragen erstellt...",
    "analyze_followup": "Nachfolgefrage wird analysiert...",
    "process_sub_questions": "Teilfragen werden parallel verarbeitet (RAG + Websuche + Synthese)...",
    "allegation_validation": "Behauptungen werden gegen Fakten validiert...",
    "final_synthesis": "Gesamtgutachten wird erstellt...",
    "red_team": "Red-Team-Pruefung auf verpasste Fehler laeuft...",
    "followup_respond": "Antwort auf Nachfolgefrage wird erstellt...",
}

# Nodes that emit the final message content
_RESPONSE_NODES = {"respond", "followup_respond"}


def _extract_content(state_update: dict) -> str:
    """Extract plain-text content from a node's message list."""
    for m in state_update.get("messages", []):
        raw = m.content if hasattr(m, "content") else str(m)
        if isinstance(raw, list):
            raw = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw
            )
        if raw:
            return raw
    return ""


@cl.on_chat_start
async def on_chat_start():
    # Keep chat start side-effect free to avoid duplicate welcome messages
    # when the websocket reconnects during long-running analysis.
    return


@cl.on_message
async def on_message(msg: cl.Message):
    # Check for PDF attachment
    pdf_bytes = None
    pdf_filename = None

    if msg.elements:
        for element in msg.elements:
            if element.mime == "application/pdf":
                with open(element.path, "rb") as f:
                    pdf_bytes = f.read()
                pdf_filename = element.name
                break

    user_text = (msg.content or "").strip()

    # ── CASE 1: New PDF uploaded → run full analysis ──────────────────
    if pdf_bytes is not None:
        await _handle_pdf_analysis(pdf_bytes, pdf_filename, user_text)
        return

    # ── CASE 2: No PDF, but session has context → follow-up question ──
    stored_pdf_content = cl.user_session.get("pdf_content")
    if stored_pdf_content and user_text:
        await _handle_followup(user_text)
        return

    # ── CASE 3: No PDF, no session context → prompt user ─────────────
    await cl.Message(
        content="Bitte laden Sie ein PDF-Dokument hoch, um die Analyse zu starten."
    ).send()


async def _handle_pdf_analysis(pdf_bytes: bytes, pdf_filename: str, user_text: str):
    """Run the full analysis pipeline on a newly uploaded PDF."""
    user_query = user_text if user_text else None

    if user_query:
        await cl.Message(
            content=f"PDF **{pdf_filename}** empfangen. Fokussierte Analyse zu Ihrer Frage wird gestartet..."
        ).send()
    else:
        await cl.Message(
            content=f"PDF **{pdf_filename}** empfangen. Vollstaendige Analyse wird gestartet..."
        ).send()

    initial_state = {
        "messages": [
            HumanMessage(content=user_text or f"Analysiere: {pdf_filename}")
        ],
        "user_query": user_query,
        "pdf_bytes": pdf_bytes,
        "pdf_filename": pdf_filename,
        "pdf_content": None,
        "raw_text": None,
        "document_structure": {},
        "document_summary": None,
        "sub_questions": [],
        "current_sub_q_index": 0,
        "facts": [],
        "allegations": [],
        "contradictions": [],
        "validation_report": {},
        "red_team_findings": [],
        "citations": [],
        "issues_to_check": [],
        "final_analysis": None,
        "previous_analysis": None,
        "error": None,
    }

    final_content = ""
    saved_pdf_content = None
    saved_document_summary = None
    saved_facts = []
    saved_allegations = []
    saved_validation_report = {}
    saved_contradictions = []
    saved_citations = []
    saved_issues_to_check = []

    async for node_output in graph.astream(initial_state):
        for node_name, state_update in node_output.items():
            if node_name in _NODE_LABELS:
                step = cl.Step(name=_NODE_LABELS[node_name], type="tool")
                await step.send()

            # Capture intermediate state for session persistence
            if state_update.get("pdf_content"):
                saved_pdf_content = state_update["pdf_content"]
            if state_update.get("document_summary"):
                saved_document_summary = state_update["document_summary"]
            if state_update.get("facts") is not None:
                saved_facts = state_update.get("facts", [])
            if state_update.get("allegations") is not None:
                saved_allegations = state_update.get("allegations", [])
            if state_update.get("validation_report") is not None:
                saved_validation_report = state_update.get("validation_report", {})
            if state_update.get("contradictions") is not None:
                saved_contradictions = state_update.get("contradictions", [])
            if state_update.get("citations") is not None:
                saved_citations = state_update.get("citations", [])
            if state_update.get("issues_to_check") is not None:
                saved_issues_to_check = state_update.get("issues_to_check", [])

            if node_name in _RESPONSE_NODES:
                content = _extract_content(state_update)
                if content:
                    final_content = content

    if final_content:
        await cl.Message(content=final_content).send()
        # Persist analysis in session for follow-up questions
        cl.user_session.set("pdf_content", saved_pdf_content)
        cl.user_session.set("document_summary", saved_document_summary)
        cl.user_session.set("pdf_filename", pdf_filename)
        cl.user_session.set("previous_analysis", final_content)
        cl.user_session.set("facts", saved_facts)
        cl.user_session.set("allegations", saved_allegations)
        cl.user_session.set("validation_report", saved_validation_report)
        cl.user_session.set("contradictions", saved_contradictions)
        cl.user_session.set("citations", saved_citations)
        cl.user_session.set("issues_to_check", saved_issues_to_check)
    else:
        await cl.Message(content="Die Analyse konnte nicht erstellt werden.").send()


async def _handle_followup(user_text: str):
    """Run the follow-up pipeline using the session's stored analysis context."""
    pdf_filename = cl.user_session.get("pdf_filename", "PDF")

    await cl.Message(
        content=f"Nachfolgefrage zum Dokument **{pdf_filename}** wird bearbeitet..."
    ).send()

    followup_state = {
        "messages": [HumanMessage(content=user_text)],
        "user_query": user_text,
        "pdf_bytes": None,
        "pdf_filename": pdf_filename,
        "pdf_content": cl.user_session.get("pdf_content"),
        "raw_text": cl.user_session.get("pdf_content"),
        "document_structure": {},
        "document_summary": cl.user_session.get("document_summary"),
        "sub_questions": [],
        "current_sub_q_index": 0,
        "facts": cl.user_session.get("facts", []),
        "allegations": cl.user_session.get("allegations", []),
        "contradictions": cl.user_session.get("contradictions", []),
        "validation_report": cl.user_session.get("validation_report", {}),
        "red_team_findings": [],
        "citations": cl.user_session.get("citations", []),
        "issues_to_check": cl.user_session.get("issues_to_check", []),
        "final_analysis": None,
        "previous_analysis": cl.user_session.get("previous_analysis"),
        "error": None,
    }

    final_content = ""

    async for node_output in followup_graph.astream(followup_state):
        for node_name, state_update in node_output.items():
            if node_name in _NODE_LABELS:
                step = cl.Step(name=_NODE_LABELS[node_name], type="tool")
                await step.send()

            if node_name in _RESPONSE_NODES:
                content = _extract_content(state_update)
                if content:
                    final_content = content

    if final_content:
        await cl.Message(content=final_content).send()
        # Update previous_analysis so chained follow-ups see latest context
        cl.user_session.set("previous_analysis", final_content)
    else:
        await cl.Message(
            content="Die Nachfolgefrage konnte nicht beantwortet werden."
        ).send()
