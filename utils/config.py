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
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "gemini-2.5-flash")
