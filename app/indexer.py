from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys

from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores import Chroma

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.ingest.pipeline import build_vector_store, ingest_document
from app.ingest.sources import (
    default_wiki_source,
    discover_wiki_urls,
    discord_documents_from_export,
    wiki_document_from_url,
    youtube_document_from_transcript,
)
from app.ingest.storage import get_db_connection

load_dotenv(find_dotenv())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index wiki pages and Discord exports into sqlite+chroma")
    parser.add_argument(
        "--wiki-url",
        action="append",
        default=[],
        help="Wiki page URL to ingest. Repeat for multiple pages.",
    )
    parser.add_argument(
        "--wiki-seed",
        action="append",
        default=[],
        help="Seed wiki URL for recursive discovery (same-domain wiki pages only).",
    )
    parser.add_argument(
        "--wiki-max-depth",
        type=int,
        default=1,
        help="Maximum crawl depth from each --wiki-seed URL.",
    )
    parser.add_argument(
        "--wiki-max-pages",
        type=int,
        default=200,
        help="Maximum discovered wiki pages across all seeds.",
    )
    parser.add_argument(
        "--discord-export",
        action="append",
        default=[],
        help="Path to a Discord JSON export file.",
    )
    parser.add_argument(
        "--youtube-transcript",
        action="append",
        default=[],
        help="Path to a YouTube transcript JSON file.",
    )
    return parser.parse_args()


def _ingest_wiki_urls(
    conn: sqlite3.Connection,
    vector_store: Chroma,
    wiki_urls: list[str],
) -> tuple[int, int]:
    indexed_count = 0
    skipped_count = 0

    source_cfg = default_wiki_source()
    for wiki_url in wiki_urls:
        doc = wiki_document_from_url(wiki_url, allowed_base_url=source_cfg.base_url)
        if doc is None:
            skipped_count += 1
            continue

        try:
            changed = ingest_document(conn, vector_store, source_cfg, doc)
        except Exception as exc:
            print(f"Failed to ingest wiki url {wiki_url}: {exc}")
            skipped_count += 1
            continue

        if changed:
            indexed_count += 1
        else:
            skipped_count += 1

    return indexed_count, skipped_count


def _merge_unique_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for url in urls:
        clean = url.strip()
        if not clean:
            continue
        if clean in seen:
            continue
        seen.add(clean)
        merged.append(clean)
    return merged


def _ingest_discord_exports(
    conn: sqlite3.Connection,
    vector_store: Chroma,
    paths: list[pathlib.Path],
) -> tuple[int, int]:
    indexed_count = 0
    skipped_count = 0

    for export_path in paths:
        if not export_path.exists():
            print(f"Discord export file not found: {export_path}")
            skipped_count += 1
            continue

        try:
            source_cfg, docs = discord_documents_from_export(export_path)
        except Exception as exc:
            print(f"Failed to parse Discord export {export_path}: {exc}")
            skipped_count += 1
            continue

        if not docs:
            print(f"No indexable messages in Discord export: {export_path}")
            skipped_count += 1
            continue

        for doc in docs:
            try:
                changed = ingest_document(conn, vector_store, source_cfg, doc)
            except Exception as exc:
                print(f"Failed to ingest Discord message {doc.canonical_uri}: {exc}")
                skipped_count += 1
                continue

            if changed:
                indexed_count += 1
            else:
                skipped_count += 1

    return indexed_count, skipped_count


def _ingest_youtube_transcripts(
    conn: sqlite3.Connection,
    vector_store: Chroma,
    paths: list[pathlib.Path],
) -> tuple[int, int]:
    indexed_count = 0
    skipped_count = 0

    for transcript_path in paths:
        if not transcript_path.exists():
            print(f"YouTube transcript file not found: {transcript_path}")
            skipped_count += 1
            continue

        try:
            source_cfg, doc = youtube_document_from_transcript(transcript_path)
        except Exception as exc:
            print(f"Failed to parse YouTube transcript {transcript_path}: {exc}")
            skipped_count += 1
            continue

        try:
            changed = ingest_document(conn, vector_store, source_cfg, doc)
        except Exception as exc:
            print(f"Failed to ingest YouTube transcript {doc.canonical_uri}: {exc}")
            skipped_count += 1
            continue

        if changed:
            indexed_count += 1
        else:
            skipped_count += 1

    return indexed_count, skipped_count


def main() -> None:
    args = parse_args()

    wiki_urls = list(args.wiki_url)
    wiki_seeds = list(args.wiki_seed)
    discord_exports = [pathlib.Path(path) for path in args.discord_export]
    youtube_transcripts = [pathlib.Path(path) for path in args.youtube_transcript]

    if wiki_seeds:
        try:
            discovered_urls = discover_wiki_urls(
                seed_urls=wiki_seeds,
                max_depth=args.wiki_max_depth,
                max_pages=args.wiki_max_pages,
                allowed_base_url=default_wiki_source().base_url,
            )
        except Exception as exc:
            print(f"Failed to discover wiki links from seeds: {exc}")
            discovered_urls = []

        if discovered_urls:
            print(
                f"Discovered {len(discovered_urls)} wiki page(s) "
                f"from {len(wiki_seeds)} seed URL(s)."
            )
        wiki_urls = _merge_unique_urls(wiki_urls + discovered_urls)

    if not wiki_urls and not wiki_seeds and not discord_exports and not youtube_transcripts:
        wiki_urls = ["https://calamitymod.wiki.gg/wiki/Supreme_Calamitas"]

    conn = get_db_connection()
    vector_store = build_vector_store()

    indexed_total = 0
    skipped_total = 0
    try:
        indexed, skipped = _ingest_wiki_urls(conn, vector_store, wiki_urls)
        indexed_total += indexed
        skipped_total += skipped

        indexed, skipped = _ingest_discord_exports(conn, vector_store, discord_exports)
        indexed_total += indexed
        skipped_total += skipped

        indexed, skipped = _ingest_youtube_transcripts(conn, vector_store, youtube_transcripts)
        indexed_total += indexed
        skipped_total += skipped
    finally:
        conn.close()

    print(f"Done. indexed={indexed_total}, skipped={skipped_total}")


if __name__ == "__main__":
    main()
