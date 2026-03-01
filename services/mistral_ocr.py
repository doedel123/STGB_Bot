import base64

from mistralai import Mistral

from utils.config import MISTRAL_API_KEY

_client = Mistral(api_key=MISTRAL_API_KEY)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using Mistral OCR. Returns markdown."""
    b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = _client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64}",
        },
        include_image_base64=False,
    )

    return "\n\n".join(page.markdown for page in response.pages)
