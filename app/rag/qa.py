from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from openai import NOT_GIVEN

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ..settings import (
    LLMConfig,
    DEFAULT_MAX_CONTEXT_CHARS,
    get_int_env,
    load_llm_config,
    openrouter_api_key,
)
from .query import RetrievalFilters, add_retrieval_args, filters_from_args, metadata_mods
from .retriever import retrieve

load_dotenv(find_dotenv())


def get_client(config: Optional[LLMConfig] = None) -> OpenAI:
    config = config or load_llm_config()
    if config.base_url:
        return OpenAI(
            api_key=openrouter_api_key(),
            base_url=config.base_url,
            default_headers=config.default_headers or None,
        )
    return OpenAI(api_key=openrouter_api_key())


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


def build_completion_request(
    question: str,
    docs,
    filters: Optional[RetrievalFilters] = None,
    llm_config: Optional[LLMConfig] = None,
) -> dict:
    llm_config = llm_config or load_llm_config()
    max_chars = get_int_env(
        "RAG_MAX_CONTEXT_CHARS",
        DEFAULT_MAX_CONTEXT_CHARS,
        minimum=1,
    )
    context = _build_context(docs, max_chars=max_chars)
    scope = filters.prompt_scope() if filters else None

    system_prompt = (
        "You are a game and mod assistant using retrieved context. "
        "Answer using only the supplied context. If the context is incomplete or conflicting, say so explicitly. "
        "Do not invent mechanics, numbers, or steps that are not supported by the context. "
        "Always cite sources as [1], [2], etc in your answer."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        + (
            f"Requested scope:\n{scope}\n\n"
            if scope
            else ""
        )
        + f"Retrieved context:\n{context}"
    )
    return {
        "model": llm_config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": llm_config.temperature,
        "extra_body": llm_config.extra_body or NOT_GIVEN,
    }


def answer_question_details(
    question: str,
    k: int,
    filters: Optional[RetrievalFilters] = None,
) -> dict:
    filters = filters or RetrievalFilters()
    docs = retrieve(query=question, k=k, filters=filters)
    if not docs:
        return {
            "answer": "I could not find relevant context in the knowledge base yet.",
            "sources": [],
            "filters": filters.to_dict(),
        }

    llm_config = load_llm_config()
    client = get_client(llm_config)
    response = client.chat.completions.create(
        **build_completion_request(question, docs, filters=filters, llm_config=llm_config)
    )
    return {
        "answer": (response.choices[0].message.content or "").strip(),
        "sources": [_serialize_doc(doc, index) for index, doc in enumerate(docs, start=1)],
        "filters": filters.to_dict(),
    }


def answer_question(
    question: str,
    k: int,
    filters: Optional[RetrievalFilters] = None,
) -> str:
    return answer_question_details(question, k=k, filters=filters)["answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple RAG QA")
    add_retrieval_args(parser, query_arg_name="question", query_help="User question")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filters = filters_from_args(args)
    answer = answer_question(args.question, k=args.k, filters=filters)
    print(answer)


if __name__ == "__main__":
    main()
