from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class LLMConfig:
    base_url: Optional[str]
    model: str
    temperature: float
    default_headers: dict[str, str]
    extra_body: dict[str, object] | None


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


def llm_api_key() -> str:
    api_key = (os.getenv("RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing LLM API key. Set RAG_LLM_API_KEY (or OPENAI_API_KEY) in .env."
        )
    return api_key


def load_llm_config() -> LLMConfig:
    base_url = normalize_base_url(os.getenv("RAG_LLM_BASE_URL"))
    default_headers: dict[str, str] = {}
    extra_body: dict[str, object] | None = None

    if is_openrouter_base_url(base_url):
        http_referer = (os.getenv("RAG_LLM_HTTP_REFERER") or "").strip()
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer

        app_name = (os.getenv("RAG_LLM_APP_NAME") or DEFAULT_APP_NAME).strip()
        if app_name:
            default_headers["X-Title"] = app_name

        fallback_models = parse_csv_env("RAG_LLM_FALLBACK_MODELS")
        if fallback_models:
            extra_body = {"models": fallback_models}

    model = (os.getenv("RAG_LLM_MODEL") or default_llm_model(base_url)).strip()
    temperature = get_float_env("RAG_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)

    return LLMConfig(
        base_url=base_url,
        model=model,
        temperature=temperature,
        default_headers=default_headers,
        extra_body=extra_body,
    )
