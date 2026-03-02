from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SourceConfig:
    source_key: str
    source_type: str
    game: str
    mod: Optional[str]
    base_url: str


@dataclass(frozen=True)
class ChunkPayload:
    text: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


@dataclass(frozen=True)
class IngestDocument:
    content_type: str
    canonical_uri: str
    title: Optional[str]
    external_id: Optional[str]
    language: Optional[str]
    text: str
    metadata: dict[str, Any]
    chunk_overrides: Optional[list[ChunkPayload]] = None
