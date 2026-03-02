from __future__ import annotations

import argparse
import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(find_dotenv())

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "models/text-embedding-004"


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


def retrieve(
    query: str,
    k: int = 6,
    source_type: Optional[str] = None,
) -> list[Document]:
    vector_store = get_vector_store()
    if source_type:
        return vector_store.similarity_search(query, k=k, filter={"source_type": source_type})
    return vector_store.similarity_search(query, k=k)


def _source_label(metadata: dict) -> str:
    source_type = metadata.get("source_type") or "unknown"
    canonical_uri = metadata.get("canonical_uri") or "unknown"
    author_name = metadata.get("author_name")
    if author_name:
        return f"{source_type} ({author_name}) - {canonical_uri}"
    return f"{source_type} - {canonical_uri}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve chunks from Chroma")
    parser.add_argument("query", help="Question or search query")
    parser.add_argument("--k", type=int, default=6, help="Number of chunks to return")
    parser.add_argument(
        "--source-type",
        choices=["wiki", "discord", "youtube"],
        default=None,
        help="Optional source filter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = retrieve(query=args.query, k=args.k, source_type=args.source_type)

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
