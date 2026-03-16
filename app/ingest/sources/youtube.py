from __future__ import annotations

import json
import pathlib
from typing import Any

from ..models import ChunkPayload, IngestDocument, SourceConfig
from .common import coerce_float, first_present, slugify


def youtube_document_from_transcript(path: pathlib.Path) -> tuple[SourceConfig, IngestDocument]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {"segments": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"YouTube transcript must be a JSON object or list: {path}")

    video_obj = raw.get("video")
    video_obj = video_obj if isinstance(video_obj, dict) else {}

    video_id = str(raw.get("video_id") or raw.get("id") or video_obj.get("id") or "").strip()
    title = str(raw.get("title") or video_obj.get("title") or "").strip() or path.stem
    channel_name = str(
        raw.get("channel")
        or raw.get("channel_name")
        or video_obj.get("channel")
        or video_obj.get("channel_name")
        or ""
    ).strip()
    published_at = str(
        raw.get("published_at")
        or raw.get("upload_date")
        or video_obj.get("published_at")
        or video_obj.get("upload_date")
        or ""
    ).strip() or None
    game_name = str(raw.get("game") or video_obj.get("game") or "").strip() or "Unknown"
    mod_name = str(raw.get("mod") or video_obj.get("mod") or "").strip() or None
    canonical_uri = str(
        raw.get("url")
        or raw.get("video_url")
        or raw.get("webpage_url")
        or video_obj.get("url")
        or video_obj.get("webpage_url")
        or ""
    ).strip()

    if not canonical_uri and video_id:
        canonical_uri = f"https://www.youtube.com/watch?v={video_id}"
    if not canonical_uri:
        canonical_uri = f"youtube://transcript/{slugify(path.stem) or 'unknown'}"

    segments: Any = None
    for key in ("segments", "transcript", "entries"):
        candidate = raw.get(key)
        if isinstance(candidate, list):
            segments = candidate
            break

    chunk_overrides: list[ChunkPayload] = []
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, str):
                text = segment.strip()
                if text:
                    chunk_overrides.append(ChunkPayload(text=text))
                continue
            if not isinstance(segment, dict):
                continue

            text = str(
                segment.get("text")
                or segment.get("content")
                or segment.get("snippet")
                or ""
            ).strip()
            if not text:
                continue

            start_sec = coerce_float(
                first_present(
                    segment.get("start"),
                    segment.get("start_sec"),
                    segment.get("start_time"),
                )
            )
            end_sec = coerce_float(
                first_present(
                    segment.get("end"),
                    segment.get("end_sec"),
                    segment.get("end_time"),
                )
            )
            duration = coerce_float(first_present(segment.get("duration"), segment.get("dur")))
            if end_sec is None and start_sec is not None and duration is not None:
                end_sec = start_sec + duration

            chunk_overrides.append(
                ChunkPayload(text=text, start_sec=start_sec, end_sec=end_sec)
            )

    fallback_text = str(raw.get("text") or raw.get("transcript_text") or "").strip()
    if not chunk_overrides and not fallback_text:
        raise ValueError(
            "No transcript text found. Expected `segments`/`transcript` list or a `text` field."
        )

    source_cfg = SourceConfig(
        source_key=f"youtube_{slugify(channel_name) or 'unknown'}",
        source_type="youtube",
        game=game_name,
        mod=mod_name,
        base_url="https://www.youtube.com",
    )

    if chunk_overrides:
        full_text = "\n".join(chunk.text for chunk in chunk_overrides)
    else:
        full_text = fallback_text

    ingest_doc = IngestDocument(
        content_type="youtube_transcript",
        canonical_uri=canonical_uri,
        title=title,
        external_id=video_id or None,
        language="en",
        text=full_text,
        metadata={
            "video_id": video_id or None,
            "channel_name": channel_name or None,
            "published_at": published_at,
            "transcript_file": str(path),
        },
        chunk_overrides=chunk_overrides if chunk_overrides else None,
    )
    return source_cfg, ingest_doc
