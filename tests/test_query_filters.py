from __future__ import annotations

import argparse
import sqlite3
import unittest

from app.db_init import SCHEMA
from app.ingest.storage import list_available_filters
from app.rag.query import RetrievalFilters, add_retrieval_args, filters_from_args


class QueryFilterTests(unittest.TestCase):
    def test_selected_mods_include_base_game_by_default(self) -> None:
        filters = RetrievalFilters.from_inputs(game="Terraria", mods=["Calamity"])

        self.assertTrue(filters.matches_metadata({"game": "Terraria", "mod": None}))
        self.assertTrue(filters.matches_metadata({"game": "Terraria", "mod": "Calamity"}))
        self.assertFalse(filters.matches_metadata({"game": "Terraria", "mod": "Infernum"}))

    def test_selected_mods_can_exclude_base_game(self) -> None:
        filters = RetrievalFilters.from_inputs(
            game="Terraria",
            mods=["Calamity", "Infernum"],
            include_base_game=False,
        )

        self.assertFalse(filters.matches_metadata({"game": "Terraria", "mod": None}))
        self.assertTrue(filters.matches_metadata({"game": "Terraria", "mod": "Infernum"}))

    def test_vector_store_filter_combines_supported_fields(self) -> None:
        filters = RetrievalFilters.from_inputs(game="Terraria", source_type="wiki")

        self.assertEqual(
            {"game": "Terraria", "source_type": "wiki"},
            filters.vector_store_filter(),
        )

    def test_filters_from_args_uses_shared_cli_parsing(self) -> None:
        parser = argparse.ArgumentParser()
        add_retrieval_args(parser)

        args = parser.parse_args(
            [
                "best summon build",
                "--game",
                "Terraria",
                "--mod",
                "Calamity",
                "--source-type",
                "wiki",
                "--exclude-base-game",
            ]
        )

        self.assertEqual(
            RetrievalFilters.from_inputs(
                game="Terraria",
                mods=["Calamity"],
                include_base_game=False,
                source_type="wiki",
            ),
            filters_from_args(args),
        )

    def test_list_available_filters_groups_mods_by_game(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO sources (source_key, source_type, game, mod, base_url, created_at)
            VALUES
                ('terraria_base', 'wiki', 'Terraria', NULL, 'https://example.com', 1),
                ('terraria_calamity', 'wiki', 'Terraria', 'Calamity', 'https://example.com', 1),
                ('minecraft_modded', 'wiki', 'Minecraft', 'Create', 'https://example.com', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO documents (
                document_id, source_id, content_type, external_id, canonical_uri, title, language, created_at
            )
            VALUES
                (1, 1, 'wiki_page', NULL, 'https://example.com/base', 'Base', 'en', 1),
                (2, 2, 'wiki_page', NULL, 'https://example.com/calamity', 'Calamity', 'en', 1),
                (3, 3, 'wiki_page', NULL, 'https://example.com/create', 'Create', 'en', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO document_versions (version_id, document_id, content_hash, fetched_at, is_current)
            VALUES
                (1, 1, 'a', 1, 1),
                (2, 2, 'b', 1, 1),
                (3, 3, 'c', 1, 1)
            """
        )
        conn.execute(
            """
            INSERT INTO chunks (
                version_id, chunk_index, text_content, word_count, start_sec, end_sec, metadata_json
            )
            VALUES
                (1, 0, 'base', 1, NULL, NULL, '{}'),
                (2, 0, 'calamity', 1, NULL, NULL, '{"mods": ["Calamity"]}'),
                (3, 0, 'create', 1, NULL, NULL, '{"mods": ["Create"]}')
            """
        )

        filters = list_available_filters(conn)
        self.assertEqual(
            [
                {"game": "Minecraft", "mods": ["Create"]},
                {"game": "Terraria", "mods": ["Calamity"]},
            ],
            filters,
        )
        conn.close()


if __name__ == "__main__":
    unittest.main()
