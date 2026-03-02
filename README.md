# calamity-bot

## Quick setup (macOS + VS Code)

1. Clone and switch branch:
   ```bash
   git clone https://github.com/iqnemo/calamity-bot.git
   cd calamity-bot
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
.venv/bin/python app/db_init.py

# run indexer against seed URLs
.venv/bin/python app/indexer.py

# run indexer with specific wiki pages
.venv/bin/python app/indexer.py --wiki-url "https://calamitymod.wiki.gg/wiki/Yharon,_Dragon_of_Rebirth"

# ingest Discord export JSON (e.g. from DiscordChatExporter)
.venv/bin/python app/indexer.py --discord-export data/discord/calamity_general.json

# ingest YouTube transcript JSON
.venv/bin/python app/indexer.py --youtube-transcript data/youtube/supreme_calamitas_guide.json

# retrieve top matching chunks
.venv/bin/python app/retriever.py "how does supreme calamitas phase 2 work?" --k 6

# ask the RAG pipeline a question
.venv/bin/python app/qa.py "how do i dodge supreme calamitas bullet hell?" --k 6

# quick OpenAI connectivity smoke test
.venv/bin/python smoke_test.py
```

## Discord Ingestion Notes

- The indexer supports Discord JSON exports and stores each message as `content_type=discord_message`.
- Keep exports in `data/discord/` (already ignored by git in this repo).
- Use exports from channels you are allowed to process; avoid indexing private messages without permission.
- The indexer supports YouTube transcript JSON as `content_type=youtube_transcript`.
- Preferred transcript shape:
  - Root object with optional video metadata (`video_id`, `title`, `channel`, `url`)
  - `segments` or `transcript` list where each item has `text` and optional `start`/`end` (or `duration`)
- To run QA with OpenRouter (or other OpenAI-compatible APIs), set `RAG_LLM_API_KEY`, `RAG_LLM_BASE_URL`, and `RAG_LLM_MODEL` in `.env`.
