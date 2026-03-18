from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import sqlite3
import time
from typing import Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..rag.query import normalize_mods
from .models import ChunkPayload, IngestDocument, SourceConfig
from .storage import (
    count_chunks_for_version,
    ensure_source,
    get_current_version,
    insert_new_version,
    store_chunks_in_db,
    upsert_document,
)


def _normalized_mods(source: SourceConfig, ingest_doc: IngestDocument) -> tuple[str, ...]:
    mods_value = ingest_doc.metadata.get("mods")
    if isinstance(mods_value, (list, tuple, set)):
        mods = normalize_mods(mods_value)
        if mods:
            return mods

    if source.mod:
        return (source.mod,)
    return ()


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def chunk_payloads_from_text(text: str) -> list[ChunkPayload]:
    return [ChunkPayload(text=chunk_text) for chunk_text in split_text(text)]


def _content_fingerprint(
    ingest_doc: IngestDocument,
    chunks: list[ChunkPayload],
) -> str:
    payload = {
        "content_type": ingest_doc.content_type,
        "title": ingest_doc.title,
        "external_id": ingest_doc.external_id,
        "language": ingest_doc.language,
        "text": ingest_doc.text,
        "metadata": ingest_doc.metadata,
        "chunks": [
            {
                "text": chunk.text,
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
            }
            for chunk in chunks
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _to_chroma_scalar(value: Any) -> Optional[Any]:
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


def add_chunks_to_chroma(
    vector_store: Chroma,
    source: SourceConfig,
    ingest_doc: IngestDocument,
    document_id: int,
    version_id: int,
    chunks: list[ChunkPayload],
    previous_chunk_count: int,
) -> None:
    ids: list[str] = []
    docs: list[Document] = []

    base_metadata: dict[str, Any] = {
        "source_key": source.source_key,
        "source_type": source.source_type,
        "game": source.game,
        "mod": source.mod,
        "content_type": ingest_doc.content_type,
        "canonical_uri": ingest_doc.canonical_uri,
        "title": ingest_doc.title,
        "external_id": ingest_doc.external_id,
        "language": ingest_doc.language,
        "document_id": document_id,
        "version_id": version_id,
    }
    normalized_mods = _normalized_mods(source, ingest_doc)
    base_metadata["mods_csv"] = "|".join(normalized_mods)
    base_metadata["is_base_game"] = len(normalized_mods) == 0

    for key, value in ingest_doc.metadata.items():
        scalar = _to_chroma_scalar(value)
        if scalar is not None:
            base_metadata[key] = scalar

    for idx, chunk in enumerate(chunks):
        ids.append(f"{source.source_key}:{document_id}:{idx}")
        chunk_metadata = dict(base_metadata)
        chunk_metadata["chunk_index"] = idx
        if chunk.start_sec is not None:
            chunk_metadata["start_sec"] = chunk.start_sec
        if chunk.end_sec is not None:
            chunk_metadata["end_sec"] = chunk.end_sec
        docs.append(Document(page_content=chunk.text, metadata=chunk_metadata))

    vector_store.add_documents(docs, ids=ids)

    if previous_chunk_count > len(chunks):
        stale_ids = [
            f"{source.source_key}:{document_id}:{idx}"
            for idx in range(len(chunks), previous_chunk_count)
        ]
        vector_store.delete(ids=stale_ids)


def ingest_document(
    conn: sqlite3.Connection,
    vector_store: Chroma,
    source: SourceConfig,
    ingest_doc: IngestDocument,
    *,
    status_writer: Callable[[str], None] | None = None,
) -> bool:
    text = ingest_doc.text.strip()
    if not text:
        return False

    if ingest_doc.chunk_overrides is not None:
        chunks = [chunk for chunk in ingest_doc.chunk_overrides if chunk.text.strip()]
    else:
        chunks = chunk_payloads_from_text(text)

    if not chunks:
        return False

    now_ts = int(time.time())
    try:
        source_id = ensure_source(conn, source, now_ts)
        document_id = upsert_document(
            conn=conn,
            source_id=source_id,
            canonical_uri=ingest_doc.canonical_uri,
            title=ingest_doc.title,
            content_type=ingest_doc.content_type,
            external_id=ingest_doc.external_id,
            language=ingest_doc.language,
            now_ts=now_ts,
        )

        content_hash = _content_fingerprint(ingest_doc, chunks)
        current = get_current_version(conn, document_id)
        if current and current["content_hash"] == content_hash:
            conn.commit()
            return False

        previous_chunk_count = 0
        if current is not None:
            previous_chunk_count = count_chunks_for_version(conn, int(current["version_id"]))

        version_id = insert_new_version(conn, document_id, content_hash, now_ts)
        metadata_json = json.dumps(ingest_doc.metadata, ensure_ascii=True, sort_keys=True)
        store_chunks_in_db(conn, version_id, chunks, metadata_json=metadata_json)

        add_chunks_to_chroma(
            vector_store=vector_store,
            source=source,
            ingest_doc=ingest_doc,
            document_id=document_id,
            version_id=version_id,
            chunks=chunks,
            previous_chunk_count=previous_chunk_count,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    if status_writer is not None:
        status_writer(
            f"Indexed {len(chunks)} chunks from {ingest_doc.canonical_uri} "
            f"(content_type={ingest_doc.content_type}, version_id={version_id})."
        )
    return True
