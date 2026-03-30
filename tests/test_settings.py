from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.settings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    load_embedding_config,
    load_llm_config,
)


class SettingsTests(unittest.TestCase):
    def test_load_llm_config_prefers_openrouter_env_names(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
                "OPENROUTER_FALLBACK_MODELS": "model-a,model-b",
                "RAG_CHAT_MODEL": "anthropic/claude-sonnet-4",
            },
            clear=False,
        ):
            config = load_llm_config()

        self.assertEqual("https://openrouter.ai/api/v1", config.base_url)
        self.assertEqual("anthropic/claude-sonnet-4", config.model)
        self.assertEqual({"models": ["model-a", "model-b"]}, config.extra_body)

    def test_load_embedding_config_uses_openrouter_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = load_embedding_config()

        self.assertEqual(DEFAULT_OPENROUTER_BASE_URL, config.base_url)
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, config.model)


if __name__ == "__main__":
    unittest.main()
