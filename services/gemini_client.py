import asyncio
import logging
import threading
import time

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import (
    GEMINI_API_KEY, MODEL_NAME, FALLBACK_MODEL_NAME,
    OPENAI_API_KEY, OPENAI_MODEL_NAME, OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_REASONING_EFFORT, OPENAI_VERBOSITY,
)

logger = logging.getLogger(__name__)

# ── Provider selection ───────────────────────────────────────────────
# Simple global instead of contextvars.ContextVar — ContextVar doesn't
# propagate into LangGraph's worker threads, causing the provider to
# silently fall back to "gemini".

_current_provider = "gemini"
_VALID_PROVIDERS = {"gemini", "openai"}


def set_provider(provider: str):
    """Set the active LLM provider."""
    global _current_provider
    _current_provider = provider


def get_provider() -> str:
    """Get the active LLM provider."""
    return _current_provider


def ensure_provider(provider: str | None) -> str:
    """Switch the process-wide provider if a valid explicit provider is given."""
    if provider in _VALID_PROVIDERS and provider != _current_provider:
        logger.info("Switching active LLM provider to %s", provider)
        set_provider(provider)
    return _current_provider


# ── Timeouts & circuit breaker ───────────────────────────────────────

_TIMEOUT = 120                # Max seconds to wait for the primary Gemini model
_OPENAI_TIMEOUT = 420         # Reasoning over OCR-heavy legal docs can exceed 3 min
_OPENAI_REQUEST_TIMEOUT = 30  # Keep create/retrieve requests short; we poll ourselves
_OPENAI_POLL_INTERVAL = 2.0
_OPENAI_STATUS_LOG_INTERVAL = 30.0
_COOLDOWN = 300               # Skip primary for 5 minutes after a failure

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

# ── OpenAI — direct SDK usage (bypasses LangChain's buggy Responses API) ─

_openai_sync_client = None
_openai_async_client = None


def _get_openai_sync():
    global _openai_sync_client
    if _openai_sync_client is None:
        import openai
        _openai_sync_client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=_OPENAI_REQUEST_TIMEOUT,
            max_retries=0,
        )
    return _openai_sync_client


def _get_openai_async():
    global _openai_async_client
    if _openai_async_client is None:
        import openai
        _openai_async_client = openai.AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            timeout=_OPENAI_REQUEST_TIMEOUT,
            max_retries=0,
        )
    return _openai_async_client


def _langchain_to_openai_payload(messages):
    """Convert LangChain messages to Responses API instructions + input."""
    instructions = []
    result = []
    for msg in messages:
        if hasattr(msg, "type"):
            if msg.type == "system":
                instructions.append(str(msg.content))
                continue
            elif msg.type == "human":
                role = "user"
            elif msg.type == "ai":
                role = "assistant"
            else:
                role = "user"
        else:
            role = "user"
        result.append({"role": role, "content": msg.content})
    return {
        "instructions": "\n\n".join(part for part in instructions if part).strip() or None,
        "input": result,
    }


def _parse_openai_response(response) -> AIMessage:
    """Extract text from OpenAI Responses API response → LangChain AIMessage."""
    if getattr(response, "output_text", ""):
        return AIMessage(content=response.output_text)

    parts = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    parts.append(block.text)
    return AIMessage(content="".join(parts))


def _openai_request_kwargs(messages) -> dict:
    payload = _langchain_to_openai_payload(messages)
    return {
        "model": OPENAI_MODEL_NAME,
        "input": payload["input"],
        "instructions": payload["instructions"],
        "reasoning": {"effort": OPENAI_REASONING_EFFORT, "summary": "auto"},
        "background": True,
        "store": True,
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
        "text": {"verbosity": OPENAI_VERBOSITY},
    }


def _openai_error_details(response) -> str:
    parts = []

    error = getattr(response, "error", None)
    if error:
        code = getattr(error, "code", None)
        message = getattr(error, "message", None) or str(error)
        parts.append(f"error={code}: {message}" if code else f"error={message}")

    incomplete = getattr(response, "incomplete_details", None)
    if incomplete:
        reason = getattr(incomplete, "reason", None) or str(incomplete)
        parts.append(f"incomplete={reason}")

    return "; ".join(parts)


def _raise_for_openai_terminal_state(response):
    status = getattr(response, "status", None) or "unknown"
    details = _openai_error_details(response)
    if details:
        raise RuntimeError(f"OpenAI response ended with status={status} ({details})")
    raise RuntimeError(f"OpenAI response ended with status={status}")


def _log_openai_status(prefix: str, response):
    status = getattr(response, "status", None) or "unknown"
    response_id = getattr(response, "id", None) or "unknown"
    output_len = len(getattr(response, "output_text", "") or "")
    details = _openai_error_details(response)

    usage = getattr(response, "usage", None)
    usage_parts = []
    if usage:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if input_tokens is not None:
            usage_parts.append(f"in={input_tokens}")
        if output_tokens is not None:
            usage_parts.append(f"out={output_tokens}")
        if total_tokens is not None:
            usage_parts.append(f"total={total_tokens}")

    extra = []
    if usage_parts:
        extra.append("usage " + " ".join(usage_parts))
    if output_len:
        extra.append(f"output_chars={output_len}")
    if details:
        extra.append(details)

    suffix = f" ({'; '.join(extra)})" if extra else ""
    logger.info("OpenAI response %s id=%s status=%s%s", prefix, response_id, status, suffix)


def _wait_for_openai_response_sync(client, response, deadline: float) -> AIMessage:
    current = response
    last_status = None
    last_log_at = 0.0

    while True:
        status = getattr(current, "status", None)
        now = time.monotonic()
        if status != last_status or (now - last_log_at) >= _OPENAI_STATUS_LOG_INTERVAL:
            _log_openai_status("poll", current)
            last_status = status
            last_log_at = now
        if status == "completed":
            return _parse_openai_response(current)
        if status in {"failed", "cancelled", "incomplete"}:
            _raise_for_openai_terminal_state(current)

        response_id = getattr(current, "id", None)
        if not response_id:
            raise RuntimeError(f"OpenAI response missing id while status={status or 'unknown'}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            try:
                client.responses.cancel(response_id)
            except Exception:
                logger.warning("OpenAI response cancel failed for id=%s", response_id, exc_info=True)
            raise TimeoutError(f"OpenAI invoke timed out after {_OPENAI_TIMEOUT}s")

        time.sleep(min(_OPENAI_POLL_INTERVAL, remaining))
        current = client.responses.retrieve(
            response_id,
            timeout=min(_OPENAI_REQUEST_TIMEOUT, remaining),
        )


async def _wait_for_openai_response_async(client, response, deadline: float) -> AIMessage:
    current = response
    last_status = None
    last_log_at = 0.0

    while True:
        status = getattr(current, "status", None)
        now = time.monotonic()
        if status != last_status or (now - last_log_at) >= _OPENAI_STATUS_LOG_INTERVAL:
            _log_openai_status("poll", current)
            last_status = status
            last_log_at = now
        if status == "completed":
            return _parse_openai_response(current)
        if status in {"failed", "cancelled", "incomplete"}:
            _raise_for_openai_terminal_state(current)

        response_id = getattr(current, "id", None)
        if not response_id:
            raise RuntimeError(f"OpenAI response missing id while status={status or 'unknown'}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            try:
                await client.responses.cancel(response_id)
            except Exception:
                logger.warning("OpenAI response cancel failed for id=%s", response_id, exc_info=True)
            raise TimeoutError(f"OpenAI invoke timed out after {_OPENAI_TIMEOUT}s")

        await asyncio.sleep(min(_OPENAI_POLL_INTERVAL, remaining))
        current = await client.responses.retrieve(
            response_id,
            timeout=min(_OPENAI_REQUEST_TIMEOUT, remaining),
        )


def _openai_invoke_sync(messages) -> AIMessage:
    """Call OpenAI Responses API synchronously via background job + polling."""
    client = _get_openai_sync()
    response = client.responses.create(
        **_openai_request_kwargs(messages),
        timeout=_OPENAI_REQUEST_TIMEOUT,
    )
    _log_openai_status("created", response)
    return _wait_for_openai_response_sync(
        client,
        response,
        deadline=time.monotonic() + _OPENAI_TIMEOUT,
    )


async def _openai_invoke_async(messages) -> AIMessage:
    """Call OpenAI Responses API asynchronously via background job + polling."""
    client = _get_openai_async()
    response = await client.responses.create(
        **_openai_request_kwargs(messages),
        timeout=_OPENAI_REQUEST_TIMEOUT,
    )
    _log_openai_status("created", response)
    return await _wait_for_openai_response_async(
        client,
        response,
        deadline=time.monotonic() + _OPENAI_TIMEOUT,
    )


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


def _sync_openai_with_timeout(messages, timeout):
    """Run OpenAI sync invoke in a daemon thread with a timeout."""
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = _openai_invoke_sync(messages)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"OpenAI invoke timed out after {timeout}s")
    if error[0]:
        raise error[0]
    return result[0]


# ── Unified LLM wrapper ─────────────────────────────────────────────

class _LLMWithFallback:
    """LLM wrapper that routes to Gemini (with circuit breaker) or OpenAI
    based on the ``_current_provider`` global variable.
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
        if _current_provider == "openai":
            logger.info("OpenAI invoke: sending request (timeout=%ds)...", _OPENAI_TIMEOUT)
            try:
                return _sync_openai_with_timeout(messages, timeout=_OPENAI_TIMEOUT)
            except Exception as e:
                logger.error("OpenAI invoke failed: %s", e)
                raise

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
        if _current_provider == "openai":
            logger.info("OpenAI ainvoke: sending request (timeout=%ds)...", _OPENAI_TIMEOUT)
            try:
                return await asyncio.wait_for(
                    _openai_invoke_async(messages),
                    timeout=_OPENAI_TIMEOUT,
                )
            except Exception as e:
                logger.error("OpenAI ainvoke failed: %s", e)
                raise

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
