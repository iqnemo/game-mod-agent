from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ..settings import (
    DEFAULT_FILTERED_FETCH_K,
    DEFAULT_FETCH_K_MULTIPLIER,
    DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    DEFAULT_MIN_FETCH_K,
    DEFAULT_RETRIEVAL_K,
    get_int_env,
)
from .query import RetrievalFilters, add_retrieval_args, filters_from_args, metadata_mods
from .vector_store import build_vector_store

load_dotenv(find_dotenv())


def _document_key(doc: Document) -> str:
    metadata = doc.metadata
    canonical_uri = metadata.get("canonical_uri")
    if isinstance(canonical_uri, str) and canonical_uri.strip():
        return canonical_uri.strip()

    document_id = metadata.get("document_id")
    if document_id is not None:
        return f"document:{document_id}"

    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return f"title:{title.strip()}"

    return doc.page_content


def _select_diverse_documents(
    docs: list[Document],
    limit: int,
    max_per_document: int,
) -> list[Document]:
    selected: list[Document] = []
    per_document_counts: dict[str, int] = {}
    selected_indexes: set[int] = set()

    for index, doc in enumerate(docs):
        document_key = _document_key(doc)
        count = per_document_counts.get(document_key, 0)
        if count >= max_per_document:
            continue
        selected.append(doc)
        selected_indexes.add(index)
        per_document_counts[document_key] = count + 1
        if len(selected) >= limit:
            break

    if len(selected) >= limit:
        return selected

    for index, doc in enumerate(docs):
        if index in selected_indexes:
            continue
        selected.append(doc)
        if len(selected) >= limit:
            break

    return selected


def _apply_filters(candidates: list[Document], filters: RetrievalFilters) -> list[Document]:
    return [doc for doc in candidates if filters.matches_metadata(doc.metadata)]


def retrieve(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
    filters: Optional[RetrievalFilters] = None,
) -> list[Document]:
    if k < 1:
        raise ValueError("k must be >= 1")

    filters = filters or RetrievalFilters()

    vector_store = build_vector_store(
        missing_key_message="GOOGLE_API_KEY is missing. Set it in .env before retrieval."
    )
    default_fetch_k = max(k * DEFAULT_FETCH_K_MULTIPLIER, DEFAULT_MIN_FETCH_K)
    if filters.game or filters.mods or filters.source_type:
        default_fetch_k = max(default_fetch_k, DEFAULT_FILTERED_FETCH_K)
    fetch_k = max(
        k,
        get_int_env(
            "RAG_RETRIEVAL_FETCH_K",
            default_fetch_k,
            minimum=1,
        ),
    )
    max_per_document = get_int_env(
        "RAG_MAX_CHUNKS_PER_DOCUMENT",
        DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
        minimum=1,
    )

    base_filter = filters.vector_store_filter()
    if base_filter:
        candidates = vector_store.similarity_search(query, k=fetch_k, filter=base_filter)
    else:
        candidates = vector_store.similarity_search(query, k=fetch_k)

    candidates = _apply_filters(candidates, filters)
    return _select_diverse_documents(candidates, limit=k, max_per_document=max_per_document)


def _source_label(metadata: dict) -> str:
    source_type = metadata.get("source_type") or "unknown"
    game = metadata.get("game") or "unknown"
    mods = metadata_mods(metadata)
    canonical_uri = metadata.get("canonical_uri") or "unknown"
    author_name = metadata.get("author_name")
    scope = game if not mods else f"{game} / {', '.join(mods)}"
    if author_name:
        return f"{source_type} [{scope}] ({author_name}) - {canonical_uri}"
    return f"{source_type} [{scope}] - {canonical_uri}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve chunks from Chroma")
    add_retrieval_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filters = filters_from_args(args)
    docs = retrieve(query=args.query, k=args.k, filters=filters)

    if not docs:
        print("No matches found.")
        return

    for idx, doc in enumerate(docs, start=1):
        label = _source_label(doc.metadata)
        text = doc.page_content.replace("\n", " ").strip()
        snippet = text[:260] + ("..." if len(text) > 260 else "")
        print(f"[{idx}] {label}")
        print(f"    {snippet}")


if __name__ == "__main__":
    main()
