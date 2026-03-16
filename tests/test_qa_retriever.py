from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.qa import build_completion_request
from app.query import RetrievalFilters
from app.retriever import _select_diverse_documents


class QaRetrieverTests(unittest.TestCase):
    def test_select_diverse_documents_limits_repeated_sources(self) -> None:
        docs = [
            Document(page_content="a1", metadata={"canonical_uri": "doc-a"}),
            Document(page_content="a2", metadata={"canonical_uri": "doc-a"}),
            Document(page_content="a3", metadata={"canonical_uri": "doc-a"}),
            Document(page_content="b1", metadata={"canonical_uri": "doc-b"}),
        ]

        selected = _select_diverse_documents(docs, limit=3, max_per_document=2)

        self.assertEqual(["a1", "a2", "b1"], [doc.page_content for doc in selected])

    def test_build_completion_request_keeps_whole_context_blocks(self) -> None:
        docs = [
            Document(
                page_content="first chunk",
                metadata={"source_type": "wiki", "canonical_uri": "doc-a"},
            ),
            Document(
                page_content="second chunk should not be partially included",
                metadata={"source_type": "wiki", "canonical_uri": "doc-b"},
            ),
        ]

        with patch.dict(os.environ, {"RAG_MAX_CONTEXT_CHARS": "64"}, clear=False):
            request = build_completion_request("question", docs)

        prompt = request["messages"][1]["content"]
        self.assertIn("first chunk", prompt)
        self.assertNotIn("second chunk should not be partially included", prompt)

    def test_build_completion_request_adds_openrouter_fallbacks(self) -> None:
        docs = [
            Document(
                page_content="context",
                metadata={"source_type": "wiki", "canonical_uri": "doc-a"},
            )
        ]

        with patch.dict(
            os.environ,
            {
                "RAG_LLM_BASE_URL": "https://openrouter.ai/api/v1",
                "RAG_LLM_FALLBACK_MODELS": "deepseek/deepseek-r1,openai/gpt-4o-mini",
            },
            clear=False,
        ):
            request = build_completion_request("question", docs)

        self.assertEqual("deepseek/deepseek-v3.2", request["model"])
        self.assertEqual(
            {"models": ["deepseek/deepseek-r1", "openai/gpt-4o-mini"]},
            request["extra_body"],
        )

    def test_build_completion_request_includes_filter_scope(self) -> None:
        docs = [
            Document(
                page_content="context",
                metadata={"source_type": "wiki", "canonical_uri": "doc-a", "game": "Terraria"},
            )
        ]
        filters = RetrievalFilters.from_inputs(game="Terraria", mods=["Calamity"])

        request = build_completion_request("question", docs, filters=filters)

        prompt = request["messages"][1]["content"]
        self.assertIn("Requested scope:", prompt)
        self.assertIn("game=Terraria", prompt)
        self.assertIn("mods=Calamity", prompt)


if __name__ == "__main__":
    unittest.main()
