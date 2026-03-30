from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional
from urllib.parse import urlparse

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"

DEFAULT_RETRIEVAL_K = 6
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2
DEFAULT_FETCH_K_MULTIPLIER = 4
DEFAULT_MIN_FETCH_K = 12
DEFAULT_FILTERED_FETCH_K = 24

DEFAULT_CHAT_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_LLM_TEMPERATURE = 0.2
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_APP_NAME = "game-mod-agent"
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8000


@dataclass(frozen=True)
class LLMConfig:
    base_url: Optional[str]
    model: str
    temperature: float
    default_headers: dict[str, str]
    extra_body: dict[str, object] | None


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    model: str
    default_headers: dict[str, str]


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


def openrouter_base_url() -> str:
    return normalize_base_url(os.getenv("OPENROUTER_BASE_URL")) or DEFAULT_OPENROUTER_BASE_URL


def openrouter_api_key() -> str:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY in .env."
        )
    return api_key


def openrouter_headers(base_url: Optional[str] = None) -> dict[str, str]:
    normalized_base_url = normalize_base_url(base_url) or openrouter_base_url()
    if not is_openrouter_base_url(normalized_base_url):
        return {}

    headers: dict[str, str] = {}
    http_referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
    if http_referer:
        headers["HTTP-Referer"] = http_referer

    app_name = (os.getenv("OPENROUTER_APP_NAME") or DEFAULT_APP_NAME).strip()
    if app_name:
        headers["X-OpenRouter-Title"] = app_name

    return headers


def load_llm_config() -> LLMConfig:
    base_url = openrouter_base_url()
    default_headers = openrouter_headers(base_url)
    extra_body: dict[str, object] | None = None

    if is_openrouter_base_url(base_url):
        fallback_models = parse_csv_env("OPENROUTER_FALLBACK_MODELS")
        if fallback_models:
            extra_body = {"models": fallback_models}

    model = (os.getenv("RAG_CHAT_MODEL") or DEFAULT_CHAT_MODEL).strip()
    temperature = get_float_env("RAG_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)

    return LLMConfig(
        base_url=base_url,
        model=model,
        temperature=temperature,
        default_headers=default_headers,
        extra_body=extra_body,
    )


def load_embedding_config() -> EmbeddingConfig:
    base_url = openrouter_base_url()
    model = (os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL).strip()
    return EmbeddingConfig(
        base_url=base_url,
        model=model,
        default_headers=openrouter_headers(base_url),
    )
