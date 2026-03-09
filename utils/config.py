import os
from dotenv import load_dotenv

load_dotenv(".env.local")


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val


GEMINI_API_KEY = _require("GEMINI_API_KEY")
MISTRAL_API_KEY = _require("MISTRAL_API_KEY")
RAGIE_API_KEY = _require("RAGIE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-pro-preview")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "gemini-2.5-pro")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # optional, only needed for OpenAI provider
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-2026-03-05")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "16000"))
OPENAI_VERBOSITY = os.getenv("OPENAI_VERBOSITY", "low")
