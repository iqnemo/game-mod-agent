from __future__ import annotations

import hashlib
import pathlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

DB_PATH = "index/kb.sqlite"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "models/text-embedding-004"


@dataclass(frozen=True)
class SourceConfig:
    source_key: str
    source_type: str
    game: str
    mod: Optional[str]
    base_url: str


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_source(conn: sqlite3.Connection, source: SourceConfig, now_ts: int) -> int:
    conn.execute(
        """
        INSERT OR IGNORE INTO sources (source_key, source_type, game, mod, base_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source.source_key, source.source_type, source.game, source.mod, source.base_url, now_ts),
    )
    conn.execute(
        """
        UPDATE sources
        SET source_type = ?, game = ?, mod = ?, base_url = ?
        WHERE source_key = ?
        """,
        (source.source_type, source.game, source.mod, source.base_url, source.source_key),
    )
    row = conn.execute(
        "SELECT source_id FROM sources WHERE source_key = ?",
        (source.source_key,),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to resolve source_id.")
    return int(row["source_id"])


def upsert_document(
    conn: sqlite3.Connection,
    source_id: int,
    canonical_uri: str,
    title: Optional[str],
    content_type: str,
    external_id: Optional[str],
    language: Optional[str],
    now_ts: int,
) -> int:
    conn.execute(
        """
        INSERT OR IGNORE INTO documents (
            source_id, content_type, external_id, canonical_uri, title, language, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source_id, content_type, external_id, canonical_uri, title, language, now_ts),
    )
    conn.execute(
        """
        UPDATE documents
        SET title = ?, language = ?, updated_at = ?
        WHERE source_id = ? AND canonical_uri = ?
        """,
        (title, language, now_ts, source_id, canonical_uri),
    )
    row = conn.execute(
        """
        SELECT document_id
        FROM documents
        WHERE source_id = ? AND canonical_uri = ?
        """,
        (source_id, canonical_uri),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to resolve document_id.")
    return int(row["document_id"])


def get_current_version(conn: sqlite3.Connection, document_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT version_id, content_hash
        FROM document_versions
        WHERE document_id = ? AND is_current = 1
        LIMIT 1
        """,
        (document_id,),
    ).fetchone()


def insert_new_version(
    conn: sqlite3.Connection, document_id: int, content_hash: str, now_ts: int
) -> int:
    conn.execute(
        "UPDATE document_versions SET is_current = 0 WHERE document_id = ? AND is_current = 1",
        (document_id,),
    )
    conn.execute(
        """
        INSERT INTO document_versions (document_id, content_hash, fetched_at, is_current)
        VALUES (?, ?, ?, 1)
        """,
        (document_id, content_hash, now_ts),
    )
    row = conn.execute("SELECT last_insert_rowid() AS version_id").fetchone()
    return int(row["version_id"])


def store_chunks_in_db(conn: sqlite3.Connection, version_id: int, chunks: list[str]) -> None:
    rows = []
    for idx, chunk in enumerate(chunks):
        rows.append((version_id, idx, chunk, len(chunk.split())))
    conn.executemany(
        """
        INSERT INTO chunks (version_id, chunk_index, text_content, token_count)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


def scrape_wiki_page(url: str) -> tuple[Optional[str], str]:
    print(f"Scraping: {url}")
    try:
        response = requests.get(url, headers={"User-Agent": "CalamityBot/1.0"}, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return None, ""

    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find(id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else None

    content_div = soup.find(id="mw-content-text")
    if not content_div:
        print("No page content found.")
        return title, ""

    for element in content_div.select("script, style, .mw-editsection, #toc"):
        element.decompose()

    raw_text = content_div.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    return title, clean_text


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def add_chunks_to_chroma(
    source: SourceConfig,
    canonical_uri: str,
    document_id: int,
    version_id: int,
    chunks: list[str],
) -> None:
    pathlib.Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    docs = []
    ids = []
    for idx, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_key": source.source_key,
                    "source_type": source.source_type,
                    "game": source.game,
                    "mod": source.mod,
                    "content_type": "wiki_page",
                    "canonical_uri": canonical_uri,
                    "document_id": document_id,
                    "version_id": version_id,
                    "chunk_index": idx,
                },
            )
        )
        ids.append(f"{source.source_key}:{document_id}:{version_id}:{idx}")

    vector_store.add_documents(docs, ids=ids)


def ingest_wiki_page(source: SourceConfig, url: str) -> None:
    title, text = scrape_wiki_page(url)
    if not text:
        print("No text extracted; skipping.")
        return

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = split_text(text)
    if not chunks:
        print("No chunks generated; skipping.")
        return

    now_ts = int(time.time())
    conn = get_db_connection(DB_PATH)

    try:
        source_id = ensure_source(conn, source, now_ts)
        document_id = upsert_document(
            conn=conn,
            source_id=source_id,
            canonical_uri=url,
            title=title,
            content_type="wiki_page",
            external_id=None,
            language="en",
            now_ts=now_ts,
        )

        current = get_current_version(conn, document_id)
        if current and current["content_hash"] == content_hash:
            print("No content changes detected; skipping re-index.")
            conn.commit()
            return

        version_id = insert_new_version(conn, document_id, content_hash, now_ts)
        store_chunks_in_db(conn, version_id, chunks)
        conn.commit()

    finally:
        conn.close()

    add_chunks_to_chroma(source, url, document_id, version_id, chunks)
    print(f"Indexed {len(chunks)} chunks from {url} (version_id={version_id}).")


if __name__ == "__main__":
    source_cfg = SourceConfig(
        source_key="calamity_wiki",
        source_type="wiki",
        game="Terraria",
        mod="Calamity",
        base_url="https://calamitymod.wiki.gg",
    )

    seed_urls = [
        "https://calamitymod.wiki.gg/wiki/Supreme_Calamitas",
    ]

    for page_url in seed_urls:
        ingest_wiki_page(source_cfg, page_url)
