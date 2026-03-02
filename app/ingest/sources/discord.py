from __future__ import annotations

import json
import pathlib
from typing import Any

from ingest.models import IngestDocument, SourceConfig
from ingest.sources.common import slugify


def _extract_discord_message_text(message: dict[str, Any]) -> str:
    parts: list[str] = []

    content = str(message.get("content") or "").strip()
    if content:
        parts.append(content)

    attachments = message.get("attachments")
    if isinstance(attachments, list):
        urls: list[str] = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            maybe_url = attachment.get("url") or attachment.get("proxy_url")
            if isinstance(maybe_url, str) and maybe_url.strip():
                urls.append(maybe_url.strip())
        if urls:
            parts.append("Attachments:\n" + "\n".join(urls))

    embeds = message.get("embeds")
    if isinstance(embeds, list):
        embed_lines: list[str] = []
        for embed in embeds:
            if not isinstance(embed, dict):
                continue
            for key in ("title", "description", "url"):
                value = embed.get(key)
                if isinstance(value, str) and value.strip():
                    embed_lines.append(value.strip())
            fields = embed.get("fields")
            if isinstance(fields, list):
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    name = str(field.get("name") or "").strip()
                    value = str(field.get("value") or "").strip()
                    if name and value:
                        embed_lines.append(f"{name}: {value}")
                    elif value:
                        embed_lines.append(value)
        if embed_lines:
            parts.append("Embeds:\n" + "\n".join(embed_lines))

    return "\n\n".join(parts).strip()


def discord_documents_from_export(path: pathlib.Path) -> tuple[SourceConfig, list[IngestDocument]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Discord export must be a JSON object: {path}")

    guild = raw.get("guild")
    channel = raw.get("channel")
    messages = raw.get("messages")

    guild = guild if isinstance(guild, dict) else {}
    channel = channel if isinstance(channel, dict) else {}
    messages = messages if isinstance(messages, list) else []

    guild_id = str(guild.get("id") or "").strip()
    guild_name = str(guild.get("name") or "").strip()
    channel_id = str(channel.get("id") or "").strip()
    channel_name = str(channel.get("name") or "").strip()

    source_name = guild_name or path.stem
    source_key = f"discord_{slugify(source_name) or 'unknown'}"

    if guild_id:
        base_url = f"https://discord.com/channels/{guild_id}"
    else:
        base_url = "https://discord.com/channels"

    source_cfg = SourceConfig(
        source_key=source_key,
        source_type="discord",
        game="Terraria",
        mod="Calamity",
        base_url=base_url,
    )

    docs: list[IngestDocument] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        text = _extract_discord_message_text(message)
        if not text:
            continue

        message_id = str(message.get("id") or "").strip()
        timestamp = str(message.get("timestamp") or "").strip() or None

        author = message.get("author")
        author = author if isinstance(author, dict) else {}
        author_name = str(author.get("name") or "").strip() or "unknown"
        author_id = str(author.get("id") or "").strip() or None

        if message_id and guild_id and channel_id:
            canonical_uri = f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
        elif message_id and channel_id:
            canonical_uri = f"discord://{channel_id}/{message_id}"
        elif message_id:
            canonical_uri = f"discord://message/{message_id}"
        else:
            canonical_uri = f"discord://{slugify(path.stem)}/row-{idx}"

        title_channel = f"#{channel_name}" if channel_name else "discord"
        title = f"{title_channel} message by {author_name}"

        metadata = {
            "guild_id": guild_id or None,
            "guild_name": guild_name or None,
            "channel_id": channel_id or None,
            "channel_name": channel_name or None,
            "author_id": author_id,
            "author_name": author_name,
            "timestamp": timestamp,
            "export_file": str(path),
        }

        docs.append(
            IngestDocument(
                content_type="discord_message",
                canonical_uri=canonical_uri,
                title=title,
                external_id=message_id or None,
                language="en",
                text=text,
                metadata=metadata,
            )
        )

    return source_cfg, docs
