from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import GEMINI_API_KEY, MODEL_NAME

# LangChain-wrapped model for use in LangGraph nodes
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)


def extract_text(response: BaseMessage) -> str:
    """Extract plain text from a LangChain response.

    Gemini sometimes returns response.content as a list of content blocks
    like [{'type': 'text', 'text': '...', 'extras': {...}}] instead of a
    plain string.  This helper normalises both forms into a single string.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)

# Direct google-genai client for search grounding (exposes grounding_metadata)
_genai_client = genai.Client(api_key=GEMINI_API_KEY)


def search_with_grounding(query: str) -> dict:
    """Call Gemini with Google Search grounding enabled.

    Returns dict with 'text' and 'sources' (list of {uri, title}).
    """
    response = _genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=query,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        ),
    )

    sources = []
    candidate = response.candidates[0] if response.candidates else None
    if candidate and candidate.grounding_metadata:
        meta = candidate.grounding_metadata
        if meta.grounding_chunks:
            for chunk in meta.grounding_chunks:
                if chunk.web:
                    sources.append({"uri": chunk.web.uri, "title": chunk.web.title})

    return {"text": response.text, "sources": sources}
