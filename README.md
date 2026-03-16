# game-mod-agent

## Quick setup (macOS + VS Code)

1. Clone and switch branch:
   ```bash
   git clone https://github.com/iqnemo/game-mod-agent.git
   cd game-mod-agent
   git checkout feat/scraper-skeleton
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Set environment variables:
   ```bash
   cp env.example .env
   ```
   Fill in `.env` with:
   - `GOOGLE_API_KEY` (used by embeddings/indexing)
   - `OPENAI_API_KEY` (used by `smoke_test.py`)
5. Open in VS Code:
   ```bash
   code .
   ```

## Useful commands

```bash
# initialize sqlite schema
.venv/bin/python -m app.db_init

# run indexer against seed URLs
.venv/bin/python -m app.indexer

# run indexer with specific wiki pages
.venv/bin/python -m app.indexer --wiki-url "https://calamitymod.wiki.gg/wiki/Yharon,_Dragon_of_Rebirth"

# discover wiki links from a guide page (1 hop, up to 200 pages by default)
.venv/bin/python -m app.indexer --wiki-seed "https://calamitymod.wiki.gg/wiki/Guide:Class_setups/Post-Moon_Lord"

# tune recursive crawl budget
.venv/bin/python -m app.indexer --wiki-seed "https://calamitymod.wiki.gg/wiki/Guide:Class_setups/Post-Moon_Lord" --wiki-max-depth 2 --wiki-max-pages 400

# ingest Discord export JSON (e.g. from DiscordChatExporter)
.venv/bin/python -m app.indexer --discord-export data/discord/calamity_general.json

# ingest YouTube transcript JSON
.venv/bin/python -m app.indexer --youtube-transcript data/youtube/supreme_calamitas_guide.json

# retrieve top matching chunks
.venv/bin/python -m app.retriever "how does supreme calamitas phase 2 work?" --k 6

# ask the RAG pipeline a question
.venv/bin/python -m app.qa "how do i dodge supreme calamitas bullet hell?" --k 6

# quick OpenAI connectivity smoke test
.venv/bin/python smoke_test.py

# run regression tests
.venv/bin/python -m unittest discover -s tests
```

## Ingestion Notes

- Wiki crawling is opt-in via `--wiki-seed` and is limited to same-domain `/wiki/...` pages.
- The crawler skips non-content namespaces such as `Special:`, `File:`, `Category:`, `Template:`, and `User:`.
- The indexer supports Discord JSON exports and stores each message as `content_type=discord_message`.
- Keep exports in `data/discord/` (already ignored by git in this repo).
- Use exports from channels you are allowed to process; avoid indexing private messages without permission.
- The indexer supports YouTube transcript JSON as `content_type=youtube_transcript`.
- Preferred transcript shape:
  - Root object with optional video metadata (`video_id`, `title`, `channel`, `url`)
  - `segments` or `transcript` list where each item has `text` and optional `start`/`end` (or `duration`)
- To run QA with OpenRouter (or other OpenAI-compatible APIs), set `RAG_LLM_API_KEY`, `RAG_LLM_BASE_URL`, and `RAG_LLM_MODEL` in `.env`.
