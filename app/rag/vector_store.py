from __future__ import annotations

import pathlib
import threading

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from ..settings import CHROMA_DIR, COLLECTION_NAME, load_embedding_config, openrouter_api_key

_lock = threading.Lock()
_cached_store: Chroma | None = None


class OpenRouterEmbeddings(Embeddings):
    def __init__(self) -> None:
        config = load_embedding_config()
        self._model = config.model
        self._client = OpenAI(
            api_key=openrouter_api_key(),
            base_url=config.base_url,
            default_headers=config.default_headers or None,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
        )
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
            encoding_format="float",
        )
        return list(response.data[0].embedding)


def build_vector_store(*, missing_key_message: str) -> Chroma:
    global _cached_store

    if _cached_store is not None:
        return _cached_store

    with _lock:
        if _cached_store is not None:
            return _cached_store

        try:
            provider_api_key = openrouter_api_key()
        except RuntimeError as exc:
            raise RuntimeError(missing_key_message) from exc
        if provider_api_key.startswith("your_"):
            raise RuntimeError(missing_key_message)

        pathlib.Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        embeddings = OpenRouterEmbeddings()
        _cached_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        return _cached_store
