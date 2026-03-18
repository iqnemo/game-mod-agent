from __future__ import annotations

from contextlib import redirect_stdout
import io
import sqlite3
import unittest

from app.db_init import SCHEMA
from app.ingest import IngestDocument, SourceConfig
from app.ingest.models import ChunkPayload
from app.ingest.pipeline import ingest_document
from app.ingest.sources.wiki import discover_wiki_urls


class FakeVectorStore:
    def __init__(self) -> None:
        self.add_calls: list[tuple[list[str], list[dict[str, object]]]] = []
        self.delete_calls: list[list[str]] = []

    def add_documents(self, docs, ids):
        self.add_calls.append((list(ids), [dict(doc.metadata) for doc in docs]))

    def delete(self, ids):
        self.delete_calls.append(list(ids))


class AuditRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.vector_store = FakeVectorStore()
        self.source = SourceConfig(
            source_key="youtube_test",
            source_type="youtube",
            game="Terraria",
            mod="Calamity",
            base_url="https://www.youtube.com",
        )

    def tearDown(self) -> None:
        self.conn.close()

    def test_ingest_reindexes_when_chunk_metadata_changes(self) -> None:
        first_doc = IngestDocument(
            content_type="youtube_transcript",
            canonical_uri="https://www.youtube.com/watch?v=abc123",
            title="Guide",
            external_id="abc123",
            language="en",
            text="same text",
            metadata={"channel_name": "Channel A"},
            chunk_overrides=[ChunkPayload(text="same text", start_sec=0.0, end_sec=5.0)],
        )
        second_doc = IngestDocument(
            content_type="youtube_transcript",
            canonical_uri="https://www.youtube.com/watch?v=abc123",
            title="Guide",
            external_id="abc123",
            language="en",
            text="same text",
            metadata={"channel_name": "Channel B"},
            chunk_overrides=[ChunkPayload(text="same text", start_sec=10.0, end_sec=15.0)],
        )

        self.assertTrue(ingest_document(self.conn, self.vector_store, self.source, first_doc))
        self.assertTrue(ingest_document(self.conn, self.vector_store, self.source, second_doc))

        versions = list(
            self.conn.execute(
                "SELECT version_id, is_current FROM document_versions ORDER BY version_id"
            )
        )
        self.assertEqual([(1, 0), (2, 1)], [(row["version_id"], row["is_current"]) for row in versions])

    def test_ingest_document_is_quiet_without_status_writer(self) -> None:
        doc = IngestDocument(
            content_type="youtube_transcript",
            canonical_uri="https://www.youtube.com/watch?v=abc123",
            title="Guide",
            external_id="abc123",
            language="en",
            text="same text",
            metadata={"channel_name": "Channel A"},
            chunk_overrides=[ChunkPayload(text="same text", start_sec=0.0, end_sec=5.0)],
        )
        captured = io.StringIO()

        with redirect_stdout(captured):
            self.assertTrue(ingest_document(self.conn, self.vector_store, self.source, doc))

        self.assertEqual("", captured.getvalue())

    def test_discover_wiki_urls_rejects_off_domain_seeds(self) -> None:
        discovered = discover_wiki_urls(
            seed_urls=["https://example.com/wiki/Foo"],
            max_depth=0,
            max_pages=10,
            allowed_base_url="https://calamitymod.wiki.gg",
        )
        self.assertEqual([], discovered)

    def test_app_ingest_imports_as_package(self) -> None:
        import app.ingest  # noqa: F401


if __name__ == "__main__":
    unittest.main()
