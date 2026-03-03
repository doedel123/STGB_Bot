import asyncio
import logging

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import GEMINI_API_KEY, MODEL_NAME, FALLBACK_MODEL_NAME

logger = logging.getLogger(__name__)

# Max seconds to wait for the primary model before falling back (async path).
_ASYNC_TIMEOUT = 25

# ── LangChain-wrapped models ────────────────────────────────────────

_primary_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)

_fallback_llm = ChatGoogleGenerativeAI(
    model=FALLBACK_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)


class _LLMWithFallback:
    """Thin wrapper that tries the primary model and falls back on any error.

    For async calls (``ainvoke``) a timeout is enforced so the SDK's internal
    retries are cut short and the fallback model is tried quickly.
    """

    def invoke(self, messages, **kwargs):
        """Sync invoke — used by sequential LangGraph nodes."""
        try:
            return _primary_llm.invoke(messages, **kwargs)
        except Exception as e:
            logger.warning(
                "Primary model %s unavailable (%s). Falling back to %s.",
                MODEL_NAME, type(e).__name__, FALLBACK_MODEL_NAME,
            )
            return _fallback_llm.invoke(messages, **kwargs)

    async def ainvoke(self, messages, **kwargs):
        """Async invoke — used by parallel processing node.

        Enforces a timeout so we don't waste time on SDK-level retries
        when the primary model is overloaded.
        """
        try:
            return await asyncio.wait_for(
                _primary_llm.ainvoke(messages, **kwargs),
                timeout=_ASYNC_TIMEOUT,
            )
        except (Exception, asyncio.TimeoutError) as e:
            logger.warning(
                "Primary model %s unavailable (%s). Falling back to %s.",
                MODEL_NAME, type(e).__name__, FALLBACK_MODEL_NAME,
            )
            return await _fallback_llm.ainvoke(messages, **kwargs)


llm_with_fallback = _LLMWithFallback()


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


# ── Direct google-genai client for search grounding ──────────────────

_genai_client = genai.Client(api_key=GEMINI_API_KEY)


def search_with_grounding(query: str) -> dict:
    """Call Gemini with Google Search grounding enabled.

    Tries the primary model first; on failure falls back to the fallback model.
    Returns dict with 'text' and 'sources' (list of {uri, title}).
    """
    for model in (MODEL_NAME, FALLBACK_MODEL_NAME):
        try:
            response = _genai_client.models.generate_content(
                model=model,
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

        except Exception as e:
            if model == MODEL_NAME:
                logger.warning(
                    "Search grounding: %s failed (%s), falling back to %s",
                    MODEL_NAME, e, FALLBACK_MODEL_NAME,
                )
                continue
            raise

    raise RuntimeError("Both primary and fallback models failed")
