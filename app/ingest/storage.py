from __future__ import annotations

import json
import pathlib
import sqlite3
from typing import Optional

from .models import ChunkPayload, SourceConfig
from ..query import metadata_mods, normalize_text

DB_PATH = "index/kb.sqlite"


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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
        SET title = ?, language = ?, updated_at = ?, external_id = ?, content_type = ?
        WHERE source_id = ? AND canonical_uri = ?
        """,
        (title, language, now_ts, external_id, content_type, source_id, canonical_uri),
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


def count_chunks_for_version(conn: sqlite3.Connection, version_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS chunk_count FROM chunks WHERE version_id = ?",
        (version_id,),
    ).fetchone()
    if row is None:
        return 0
    return int(row["chunk_count"])


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
    if row is None:
        raise RuntimeError("Failed to resolve version_id.")
    return int(row["version_id"])


def store_chunks_in_db(
    conn: sqlite3.Connection,
    version_id: int,
    chunks: list[ChunkPayload],
    metadata_json: Optional[str] = None,
) -> None:
    rows = []
    for idx, chunk in enumerate(chunks):
        rows.append(
            (
                version_id,
                idx,
                chunk.text,
                len(chunk.text.split()),
                chunk.start_sec,
                chunk.end_sec,
                metadata_json,
            )
        )
    conn.executemany(
        """
        INSERT INTO chunks (
            version_id, chunk_index, text_content, token_count, start_sec, end_sec, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def list_available_filters(conn: sqlite3.Connection) -> list[dict]:
    game_to_mods: dict[str, set[str]] = {}
    rows = conn.execute(
        """
        SELECT
            s.game AS source_game,
            s.mod AS source_mod,
            c.metadata_json AS metadata_json
        FROM document_versions dv
        JOIN documents d ON d.document_id = dv.document_id
        JOIN sources s ON s.source_id = d.source_id
        LEFT JOIN chunks c ON c.version_id = dv.version_id AND c.chunk_index = 0
        WHERE dv.is_current = 1
        """
    ).fetchall()

    for row in rows:
        metadata: dict = {}
        metadata_json = row["metadata_json"]
        if isinstance(metadata_json, str) and metadata_json.strip():
            try:
                parsed = json.loads(metadata_json)
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {}

        game = normalize_text(metadata.get("game")) or normalize_text(row["source_game"])
        if game is None or game.casefold() == "unknown":
            continue

        mods = set(metadata_mods(metadata))
        source_mod = normalize_text(row["source_mod"])
        if source_mod:
            mods.add(source_mod)

        bucket = game_to_mods.setdefault(game, set())
        for mod in mods:
            if mod.casefold() == "unknown":
                continue
            bucket.add(mod)

    return [
        {"game": game, "mods": sorted(mods, key=str.casefold)}
        for game, mods in sorted(game_to_mods.items(), key=lambda item: item[0].casefold())
    ]
