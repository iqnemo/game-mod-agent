from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from openai import NOT_GIVEN

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.query import RetrievalFilters, metadata_mods
from app.retriever import retrieve
from app.settings import (
    DEFAULT_APP_NAME,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RETRIEVAL_K,
    default_llm_model,
    get_float_env,
    get_int_env,
    is_openrouter_base_url,
    normalize_base_url,
    parse_csv_env,
)

load_dotenv(find_dotenv())


def get_client() -> OpenAI:
    api_key = (os.getenv("RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing LLM API key. Set RAG_LLM_API_KEY (or OPENAI_API_KEY) in .env."
        )

    base_url = normalize_base_url(os.getenv("RAG_LLM_BASE_URL"))
    headers = _default_headers(base_url)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)
    return OpenAI(api_key=api_key)


def _default_headers(base_url: str | None) -> dict[str, str]:
    if not is_openrouter_base_url(base_url):
        return {}

    headers: dict[str, str] = {}
    http_referer = (os.getenv("RAG_LLM_HTTP_REFERER") or "").strip()
    if http_referer:
        headers["HTTP-Referer"] = http_referer

    app_name = (os.getenv("RAG_LLM_APP_NAME") or DEFAULT_APP_NAME).strip()
    if app_name:
        headers["X-Title"] = app_name

    return headers


def _format_context_header(index: int, metadata: dict) -> str:
    source_type = metadata.get("source_type") or "unknown"
    uri = metadata.get("canonical_uri") or "unknown"
    parts = [f"[{index}] source={source_type}", f"uri={uri}"]
    game = metadata.get("game")
    if game:
        parts.append(f"game={game}")

    mods = metadata_mods(metadata)
    if mods:
        parts.append(f"mods={', '.join(mods)}")

    title = metadata.get("title")
    if title:
        parts.append(f"title={title}")

    author_name = metadata.get("author_name")
    if author_name:
        parts.append(f"author={author_name}")

    channel_name = metadata.get("channel_name")
    if channel_name:
        parts.append(f"channel={channel_name}")

    timestamp = metadata.get("timestamp")
    if timestamp:
        parts.append(f"timestamp={timestamp}")

    start_sec = metadata.get("start_sec")
    end_sec = metadata.get("end_sec")
    if start_sec is not None or end_sec is not None:
        parts.append(f"time={start_sec}-{end_sec}")

    return " ".join(parts)


def _build_context(docs, max_chars: int) -> str:
    blocks: list[str] = []
    total_chars = 0

    for idx, doc in enumerate(docs, start=1):
        block = f"{_format_context_header(idx, doc.metadata)}\n{doc.page_content.strip()}"
        separator_len = 2 if blocks else 0
        projected_size = total_chars + separator_len + len(block)
        if blocks and projected_size > max_chars:
            break
        blocks.append(block)
        total_chars = projected_size

    return "\n\n".join(blocks)


def _completion_extra_body(base_url: str | None) -> dict | None:
    if not is_openrouter_base_url(base_url):
        return None

    fallback_models = parse_csv_env("RAG_LLM_FALLBACK_MODELS")
    if not fallback_models:
        return None
    return {"models": fallback_models}


def _serialize_doc(doc, index: int) -> dict:
    metadata = dict(doc.metadata)
    snippet_text = doc.page_content.replace("\n", " ").strip()
    return {
        "index": index,
        "source_type": metadata.get("source_type"),
        "canonical_uri": metadata.get("canonical_uri"),
        "title": metadata.get("title"),
        "game": metadata.get("game"),
        "mods": list(metadata_mods(metadata)),
        "author_name": metadata.get("author_name"),
        "channel_name": metadata.get("channel_name"),
        "timestamp": metadata.get("timestamp"),
        "start_sec": metadata.get("start_sec"),
        "end_sec": metadata.get("end_sec"),
        "snippet": snippet_text[:280] + ("..." if len(snippet_text) > 280 else ""),
    }


def build_completion_request(question: str, docs, filters: Optional[RetrievalFilters] = None) -> dict:
    base_url = normalize_base_url(os.getenv("RAG_LLM_BASE_URL"))
    max_chars = get_int_env(
        "RAG_MAX_CONTEXT_CHARS",
        DEFAULT_MAX_CONTEXT_CHARS,
        minimum=1,
    )
    context = _build_context(docs, max_chars=max_chars)
    model = (os.getenv("RAG_LLM_MODEL") or default_llm_model(base_url)).strip()
    temperature = get_float_env("RAG_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)

    system_prompt = (
        "You are a game and mod assistant using retrieved context. "
        "Answer using only the supplied context. If the context is incomplete or conflicting, say so explicitly. "
        "Do not invent mechanics, numbers, or steps that are not supported by the context. "
        "Always cite sources as [1], [2], etc in your answer."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        + (
            f"Requested scope:\n{filters.prompt_scope()}\n\n"
            if filters and filters.prompt_scope()
            else ""
        )
        + f"Retrieved context:\n{context}"
    )

    extra_body = _completion_extra_body(base_url)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "extra_body": extra_body or NOT_GIVEN,
    }


def answer_question_details(
    question: str,
    k: int,
    filters: Optional[RetrievalFilters] = None,
    source_type: str | None = None,
) -> dict:
    docs = retrieve(query=question, k=k, filters=filters, source_type=source_type)
    if not docs:
        return {
            "answer": "I could not find relevant context in the knowledge base yet.",
            "sources": [],
            "filters": filters.to_dict() if filters else RetrievalFilters.from_inputs(source_type=source_type).to_dict(),
        }

    client = get_client()
    response = client.chat.completions.create(
        **build_completion_request(question, docs, filters=filters)
    )
    return {
        "answer": (response.choices[0].message.content or "").strip(),
        "sources": [_serialize_doc(doc, index) for index, doc in enumerate(docs, start=1)],
        "filters": filters.to_dict() if filters else RetrievalFilters.from_inputs(source_type=source_type).to_dict(),
    }


def answer_question(
    question: str,
    k: int,
    source_type: str | None = None,
    filters: Optional[RetrievalFilters] = None,
) -> str:
    return answer_question_details(question, k=k, filters=filters, source_type=source_type)["answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple RAG QA")
    parser.add_argument("question", help="User question")
    parser.add_argument("--k", type=int, default=DEFAULT_RETRIEVAL_K, help="How many chunks to retrieve")
    parser.add_argument(
        "--source-type",
        choices=["wiki", "discord", "youtube"],
        default=None,
        help="Optional source filter",
    )
    parser.add_argument("--game", default=None, help="Optional game filter")
    parser.add_argument("--mod", action="append", default=[], help="Repeatable mod filter")
    parser.add_argument(
        "--exclude-base-game",
        action="store_true",
        help="When using --mod, exclude base game results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filters = RetrievalFilters.from_inputs(
        game=args.game,
        mods=args.mod,
        include_base_game=not args.exclude_base_game,
        source_type=args.source_type,
    )
    answer = answer_question(args.question, k=args.k, filters=filters)
    print(answer)


if __name__ == "__main__":
    main()
