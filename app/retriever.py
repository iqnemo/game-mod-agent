from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.query import RetrievalFilters, metadata_mods
from app.settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_FILTERED_FETCH_K,
    DEFAULT_FETCH_K_MULTIPLIER,
    DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    DEFAULT_MIN_FETCH_K,
    DEFAULT_RETRIEVAL_K,
    EMBEDDING_MODEL,
    get_int_env,
)

load_dotenv(find_dotenv())


def get_vector_store() -> Chroma:
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_api_key or google_api_key.startswith("your_"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Set it in .env before retrieval.")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


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


def _base_filter(filters: RetrievalFilters) -> Optional[dict]:
    if filters.game:
        return {"game": filters.game}
    if filters.source_type:
        return {"source_type": filters.source_type}
    return None


def _apply_filters(candidates: list[Document], filters: RetrievalFilters) -> list[Document]:
    return [doc for doc in candidates if filters.matches_metadata(doc.metadata)]


def retrieve(
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
    filters: Optional[RetrievalFilters] = None,
    source_type: Optional[str] = None,
) -> list[Document]:
    if k < 1:
        raise ValueError("k must be >= 1")

    if filters is None:
        filters = RetrievalFilters.from_inputs(source_type=source_type)
    elif source_type is not None and filters.source_type is None:
        filters = RetrievalFilters.from_inputs(
            game=filters.game,
            mods=filters.mods,
            include_base_game=filters.include_base_game,
            source_type=source_type,
        )

    vector_store = get_vector_store()
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

    base_filter = _base_filter(filters)
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
    parser.add_argument("query", help="Question or search query")
    parser.add_argument("--k", type=int, default=DEFAULT_RETRIEVAL_K, help="Number of chunks to return")
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
