from __future__ import annotations

import os
import pathlib
import threading

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..settings import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

_lock = threading.Lock()
_cached_store: Chroma | None = None


def build_vector_store(*, missing_key_message: str) -> Chroma:
    global _cached_store

    if _cached_store is not None:
        return _cached_store

    with _lock:
        if _cached_store is not None:
            return _cached_store

        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not google_api_key or google_api_key.startswith("your_"):
            raise RuntimeError(missing_key_message)

        pathlib.Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        _cached_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        return _cached_store
