from __future__ import annotations

import os
import pathlib

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..settings import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def build_vector_store(*, missing_key_message: str) -> Chroma:
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_api_key or google_api_key.startswith("your_"):
        raise RuntimeError(missing_key_message)

    pathlib.Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
