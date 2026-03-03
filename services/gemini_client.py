import asyncio
import logging
import threading
import time

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import GEMINI_API_KEY, MODEL_NAME, FALLBACK_MODEL_NAME

logger = logging.getLogger(__name__)

# ── Timeouts & circuit breaker ───────────────────────────────────────

_TIMEOUT = 20          # Max seconds to wait for the primary model
_COOLDOWN = 300        # Skip primary for 5 minutes after a failure

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


def _sync_invoke_with_timeout(llm, messages, kwargs, timeout):
    """Run a sync LLM invoke in a daemon thread with a timeout."""
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = llm.invoke(messages, **kwargs)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"LLM invoke timed out after {timeout}s")
    if error[0]:
        raise error[0]
    return result[0]


class _LLMWithFallback:
    """LLM wrapper with timeout, automatic fallback, and circuit breaker.

    - Tries the primary model first with a timeout.
    - On ANY failure (503, timeout, etc.) immediately switches to fallback.
    - After a failure, the primary model is skipped for ``_COOLDOWN`` seconds
      so subsequent calls go directly to the fallback without waiting.
    """

    def __init__(self):
        self._primary_down_until = 0.0

    def _primary_is_down(self) -> bool:
        if time.time() < self._primary_down_until:
            return True
        return False

    def _mark_primary_down(self, error):
        self._primary_down_until = time.time() + _COOLDOWN
        logger.warning(
            "Primary model %s unavailable (%s). "
            "Falling back to %s for the next %ds.",
            MODEL_NAME, error, FALLBACK_MODEL_NAME, _COOLDOWN,
        )

    def invoke(self, messages, **kwargs):
        """Sync invoke with timeout and circuit breaker."""
        if self._primary_is_down():
            logger.info("Primary model in cooldown — using %s.", FALLBACK_MODEL_NAME)
            return _fallback_llm.invoke(messages, **kwargs)

        try:
            return _sync_invoke_with_timeout(
                _primary_llm, messages, kwargs, timeout=_TIMEOUT,
            )
        except Exception as e:
            self._mark_primary_down(e)
            return _fallback_llm.invoke(messages, **kwargs)

    async def ainvoke(self, messages, **kwargs):
        """Async invoke with timeout and circuit breaker."""
        if self._primary_is_down():
            logger.info("Primary model in cooldown — using %s.", FALLBACK_MODEL_NAME)
            return await _fallback_llm.ainvoke(messages, **kwargs)

        try:
            return await asyncio.wait_for(
                _primary_llm.ainvoke(messages, **kwargs),
                timeout=_TIMEOUT,
            )
        except (Exception, asyncio.TimeoutError) as e:
            self._mark_primary_down(e)
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

    Uses the circuit breaker from ``llm_with_fallback`` to decide whether
    to try the primary model or skip directly to the fallback.
    """
    if llm_with_fallback._primary_is_down():
        models_to_try = (FALLBACK_MODEL_NAME,)
    else:
        models_to_try = (MODEL_NAME, FALLBACK_MODEL_NAME)

    for model in models_to_try:
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
                llm_with_fallback._mark_primary_down(e)
                continue
            raise

    raise RuntimeError("Both primary and fallback models failed")
