from __future__ import annotations

from typing import Optional

import requests
from bs4 import BeautifulSoup

from ingest.models import IngestDocument, SourceConfig


def default_wiki_source() -> SourceConfig:
    return SourceConfig(
        source_key="calamity_wiki",
        source_type="wiki",
        game="Terraria",
        mod="Calamity",
        base_url="https://calamitymod.wiki.gg",
    )


def scrape_wiki_page(url: str) -> tuple[Optional[str], str]:
    print(f"Scraping: {url}")
    try:
        response = requests.get(url, headers={"User-Agent": "CalamityBot/1.0"}, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return None, ""

    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find(id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else None

    content_div = soup.find(id="mw-content-text")
    if not content_div:
        print("No page content found.")
        return title, ""

    for element in content_div.select("script, style, .mw-editsection, #toc"):
        element.decompose()

    raw_text = content_div.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    return title, clean_text


def wiki_document_from_url(url: str) -> Optional[IngestDocument]:
    title, text = scrape_wiki_page(url)
    if not text:
        print(f"Skipping wiki url with no extracted text: {url}")
        return None

    return IngestDocument(
        content_type="wiki_page",
        canonical_uri=url,
        title=title,
        external_id=None,
        language="en",
        text=text,
        metadata={},
    )
