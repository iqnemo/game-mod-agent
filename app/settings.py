from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "models/text-embedding-004"

DEFAULT_RETRIEVAL_K = 6
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2
DEFAULT_FETCH_K_MULTIPLIER = 4
DEFAULT_MIN_FETCH_K = 12
DEFAULT_FILTERED_FETCH_K = 24

DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_APP_NAME = "game-mod-agent"
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8000


def get_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


def parse_csv_env(name: str) -> list[str]:
    raw_value = os.getenv(name, "")
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    if base_url is None:
        return None
    normalized = base_url.strip()
    return normalized or None


def is_openrouter_base_url(base_url: Optional[str]) -> bool:
    normalized = normalize_base_url(base_url)
    if normalized is None:
        return False
    parsed = urlparse(normalized)
    return parsed.netloc.lower() == "openrouter.ai"


def default_llm_model(base_url: Optional[str]) -> str:
    if is_openrouter_base_url(base_url):
        return DEFAULT_OPENROUTER_MODEL
    return DEFAULT_OPENAI_MODEL
