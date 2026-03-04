from agent.state import AgentState
from services.mistral_ocr import extract_text_from_pdf


def extract_pdf_node(state: AgentState) -> dict:
    """Extract text from the uploaded PDF using Mistral OCR."""
    pdf_bytes = state.get("pdf_bytes")
    if not pdf_bytes:
        return {"error": "Kein PDF hochgeladen."}

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        return {"error": f"PDF-Extraktion fehlgeschlagen: {e}"}

    if not text.strip():
        return {"error": "Das PDF scheint keinen extrahierbaren Text zu enthalten."}

    return {
        "pdf_content": text,
        "raw_text": text,
        "document_structure": {"source": "ocr", "has_page_blocks": False},
    }
