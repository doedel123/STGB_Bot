import asyncio
import contextvars
import logging
import threading
import time

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import (
    GEMINI_API_KEY, MODEL_NAME, FALLBACK_MODEL_NAME,
    OPENAI_API_KEY, OPENAI_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# ── Provider selection ───────────────────────────────────────────────

_current_provider = contextvars.ContextVar("llm_provider", default="gemini")


def set_provider(provider: str):
    """Set the active LLM provider for the current async context."""
    _current_provider.set(provider)


def get_provider() -> str:
    """Get the active LLM provider for the current async context."""
    return _current_provider.get()


# ── Timeouts & circuit breaker (Gemini only) ─────────────────────────

_TIMEOUT = 20          # Max seconds to wait for the primary Gemini model
_COOLDOWN = 300        # Skip primary for 5 minutes after a failure

# ── Gemini models ───────────────────────────────────────────────────

_gemini_primary = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)

_gemini_fallback = ChatGoogleGenerativeAI(
    model=FALLBACK_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)

# ── OpenAI model (lazy init — only created when first needed) ────────

_openai_llm = None


def _get_openai_llm():
    global _openai_llm
    if _openai_llm is None:
        from langchain_openai import ChatOpenAI
        _openai_llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0.2,
        )
    return _openai_llm


# ── Helpers ──────────────────────────────────────────────────────────

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


# ── Unified LLM wrapper ─────────────────────────────────────────────

class _LLMWithFallback:
    """LLM wrapper that routes to Gemini (with circuit breaker) or OpenAI
    based on the ``_current_provider`` context variable.
    """

    def __init__(self):
        self._primary_down_until = 0.0

    def _primary_is_down(self) -> bool:
        return time.time() < self._primary_down_until

    def _mark_primary_down(self, error):
        self._primary_down_until = time.time() + _COOLDOWN
        logger.warning(
            "Gemini primary %s unavailable (%s). "
            "Falling back to %s for the next %ds.",
            MODEL_NAME, error, FALLBACK_MODEL_NAME, _COOLDOWN,
        )

    # ── Sync ─────────────────────────────────────────────────────────

    def invoke(self, messages, **kwargs):
        if _current_provider.get() == "openai":
            return _get_openai_llm().invoke(messages, **kwargs)

        # Gemini path with circuit breaker
        if self._primary_is_down():
            logger.info("Gemini primary in cooldown — using %s.", FALLBACK_MODEL_NAME)
            return _gemini_fallback.invoke(messages, **kwargs)

        try:
            return _sync_invoke_with_timeout(
                _gemini_primary, messages, kwargs, timeout=_TIMEOUT,
            )
        except Exception as e:
            self._mark_primary_down(e)
            return _gemini_fallback.invoke(messages, **kwargs)

    # ── Async ────────────────────────────────────────────────────────

    async def ainvoke(self, messages, **kwargs):
        if _current_provider.get() == "openai":
            return await _get_openai_llm().ainvoke(messages, **kwargs)

        # Gemini path with circuit breaker
        if self._primary_is_down():
            logger.info("Gemini primary in cooldown — using %s.", FALLBACK_MODEL_NAME)
            return await _gemini_fallback.ainvoke(messages, **kwargs)

        try:
            return await asyncio.wait_for(
                _gemini_primary.ainvoke(messages, **kwargs),
                timeout=_TIMEOUT,
            )
        except (Exception, asyncio.TimeoutError) as e:
            self._mark_primary_down(e)
            return await _gemini_fallback.ainvoke(messages, **kwargs)


llm_with_fallback = _LLMWithFallback()


def extract_text(response: BaseMessage) -> str:
    """Extract plain text from a LangChain response.

    Works with both Gemini (may return list of content blocks) and
    OpenAI (always returns a plain string).
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
# Always uses Gemini (Google-specific feature), regardless of provider.

_genai_client = genai.Client(api_key=GEMINI_API_KEY)


def search_with_grounding(query: str) -> dict:
    """Call Gemini with Google Search grounding enabled.

    Always uses Gemini (even when OpenAI is the active provider) because
    Google Search grounding is a Gemini-specific feature.
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
