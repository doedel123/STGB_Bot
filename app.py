import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.graph import graph

# Display names for step indicators
_NODE_LABELS = {
    "extract_pdf": "PDF-Text wird extrahiert (Mistral OCR)...",
    "analyze_document": "Dokument wird analysiert und Teilfragen erstellt...",
    "retrieve_rag": "StGB/StPO-Kommentare werden abgerufen (RAGIE)...",
    "search_case_law": "Aktuelle Rechtsprechung wird gesucht (Google Search)...",
    "synthesize_step": "Zwischenanalyse wird erstellt...",
    "final_synthesis": "Gesamtgutachten wird erstellt...",
}


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "Willkommen beim **StGB-Agenten**.\n\n"
            "Laden Sie eine Anklageschrift oder ein anderes strafrechtliches "
            "Dokument als PDF hoch, und ich erstelle eine umfassende "
            "strafrechtliche Analyse im Gutachtenstil.\n\n"
            "Die Analyse nutzt:\n"
            "- StGB/StPO-Kommentarliteratur (RAGIE RAG)\n"
            "- Aktuelle Rechtsprechung (Google Search)\n"
            "- Gemini 3.1 Pro als Analyse-LLM"
        )
    ).send()


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

    if pdf_bytes is None:
        await cl.Message(
            content="Bitte laden Sie ein PDF-Dokument hoch, um die Analyse zu starten."
        ).send()
        return

    # Detect if user asked a specific question or just uploaded
    user_text = (msg.content or "").strip()
    user_query = user_text if user_text else None

    if user_query:
        await cl.Message(
            content=f"PDF **{pdf_filename}** empfangen. Fokussierte Analyse zu Ihrer Frage wird gestartet..."
        ).send()
    else:
        await cl.Message(
            content=f"PDF **{pdf_filename}** empfangen. Vollstaendige Analyse wird gestartet..."
        ).send()

    # Prepare initial state
    initial_state = {
        "messages": [
            HumanMessage(content=user_text or f"Analysiere: {pdf_filename}")
        ],
        "user_query": user_query,
        "pdf_bytes": pdf_bytes,
        "pdf_filename": pdf_filename,
        "pdf_content": None,
        "document_summary": None,
        "sub_questions": [],
        "current_sub_q_index": 0,
        "final_analysis": None,
        "error": None,
    }

    # Use astream (async) so we don't block Chainlit's event loop
    final_content = ""

    async for node_output in graph.astream(initial_state):
        # node_output is a dict like {"node_name": {state_update}}
        for node_name, state_update in node_output.items():
            # Show step indicator
            if node_name in _NODE_LABELS:
                step = cl.Step(name=_NODE_LABELS[node_name], type="tool")
                await step.send()

            # Capture the final analysis when respond node fires
            if node_name == "respond" and "messages" in state_update:
                for m in state_update["messages"]:
                    raw = m.content if hasattr(m, "content") else str(m)
                    # Gemini may return list of content blocks
                    if isinstance(raw, list):
                        raw = "".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in raw
                        )
                    if raw:
                        final_content = raw

    # Send the final analysis as a single message
    if final_content:
        await cl.Message(content=final_content).send()
    else:
        await cl.Message(content="Die Analyse konnte nicht erstellt werden.").send()
