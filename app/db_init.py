import sqlite3, pathlib

DB_PATH = "index/kb.sqlite"


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sources(
    source_id INTEGER PRIMARY KEY,
    source_key TEXT UNIQUE NOT NULL,        -- terraria_wiki, calamity_wiki, etc
    source_type TEXT NOT NULL,              -- wiki, youtube, etc.
    game TEXT NOT NULL,                     -- Terraria, Minecraft
    mod TEXT,                               -- Calamity, etc. NULL for base game
    base_url TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS documents(
    document_id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    content_type TEXT NOT NULL,             -- wiki_page, youtube_transcript, video, text
    external_id TEXT,                       -- page id, youtube video id, etc
    canonical_uri TEXT NOT NULL,            -- full URL or canonical local uri
    title TEXT,
    language TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER,
    UNIQUE(source_id, canonical_uri)
);

CREATE TABLE IF NOT EXISTS document_versions (
    version_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,               -- dedupe / change detection
    fetched_at INTEGER NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1 CHECK(is_current IN (0, 1))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    version_id INTEGER NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    token_count INTEGER,
    start_sec REAL,                           -- for transcript/video time references
    end_sec REAL,                             -- for transcript/video time references
    metadata_json TEXT,                       -- optional extra metadata
    UNIQUE(version_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_versions_document_id ON document_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_versions_current ON document_versions(document_id, is_current);
CREATE INDEX IF NOT EXISTS idx_chunks_version_id ON chunks(version_id);
"""

def init_db(db_path: str = DB_PATH) -> None:
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")