from __future__ import annotations

from collections import deque
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from ..models import IngestDocument, SourceConfig

WIKI_BLOCKED_NAMESPACES = {
    "Category",
    "File",
    "Help",
    "MediaWiki",
    "Module",
    "Special",
    "Talk",
    "Template",
    "User",
}


def default_wiki_source(
    *,
    source_key: str = "calamity_wiki",
    game: str = "Terraria",
    mod: str | None = "Calamity",
    base_url: str = "https://calamitymod.wiki.gg",
) -> SourceConfig:
    return SourceConfig(
        source_key=source_key,
        source_type="wiki",
        game=game,
        mod=mod,
        base_url=base_url,
    )


def _canonical_netloc(netloc: str) -> str:
    netloc = netloc.lower().strip()
    if netloc.startswith("www."):
        return netloc[4:]
    return netloc


def _normalize_wiki_url(url: str) -> Optional[str]:
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return None

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((parsed.scheme.lower(), _canonical_netloc(parsed.netloc), path, "", "", ""))


def _allowed_netlocs_from_base_url(allowed_base_url: Optional[str]) -> set[str]:
    if not allowed_base_url:
        return set()

    parsed_base = urlparse(allowed_base_url.strip())
    if not parsed_base.netloc:
        raise ValueError("allowed_base_url must include a hostname")
    return {_canonical_netloc(parsed_base.netloc)}


def _wiki_title_from_path(path: str) -> Optional[str]:
    if not path.startswith("/wiki/"):
        return None
    title = unquote(path[len("/wiki/") :]).strip()
    if not title:
        return None
    return title


def _is_blocked_namespace(title: str) -> bool:
    head = title.split("/", 1)[0]
    namespace = head.split(":", 1)[0]
    return namespace in WIKI_BLOCKED_NAMESPACES


def _is_allowed_wiki_url(url: str, allowed_netlocs: set[str]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if _canonical_netloc(parsed.netloc) not in allowed_netlocs:
        return False

    title = _wiki_title_from_path(parsed.path)
    if title is None:
        return False

    if _is_blocked_namespace(title):
        return False

    return True


def _fetch_wiki_html(url: str) -> Optional[str]:
    try:
        response = requests.get(url, headers={"User-Agent": "GameModAgent/1.0"}, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return None
    return response.text


def _extract_wiki_links_from_html(
    html: str,
    current_url: str,
    allowed_netlocs: set[str],
) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find(id="mw-content-text")
    if content_div is None:
        return []

    links: list[str] = []
    seen: set[str] = set()

    for anchor in content_div.select("a[href]"):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith("javascript:"):
            continue

        absolute_url = urljoin(current_url, href)
        normalized = _normalize_wiki_url(absolute_url)
        if not normalized:
            continue
        if normalized in seen:
            continue
        if not _is_allowed_wiki_url(normalized, allowed_netlocs):
            continue

        seen.add(normalized)
        links.append(normalized)

    return links


def discover_wiki_urls(
    seed_urls: list[str],
    max_depth: int = 1,
    max_pages: int = 200,
    allowed_base_url: Optional[str] = None,
) -> list[str]:
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if max_pages <= 0:
        raise ValueError("max_pages must be > 0")

    normalized_seeds = []
    for seed in seed_urls:
        normalized = _normalize_wiki_url(seed)
        if normalized:
            normalized_seeds.append(normalized)

    if not normalized_seeds:
        return []

    allowed_netlocs = _allowed_netlocs_from_base_url(allowed_base_url)
    if not allowed_netlocs:
        for seed in normalized_seeds:
            parsed_seed = urlparse(seed)
            allowed_netlocs.add(_canonical_netloc(parsed_seed.netloc))

    queue = deque(
        (seed, 0) for seed in normalized_seeds if _is_allowed_wiki_url(seed, allowed_netlocs)
    )
    seen_urls: set[str] = set()
    discovered: list[str] = []

    while queue and len(discovered) < max_pages:
        current_url, depth = queue.popleft()

        if current_url in seen_urls:
            continue
        seen_urls.add(current_url)

        if not _is_allowed_wiki_url(current_url, allowed_netlocs):
            continue

        discovered.append(current_url)

        if depth >= max_depth:
            continue

        html = _fetch_wiki_html(current_url)
        if html is None:
            continue

        for link in _extract_wiki_links_from_html(html, current_url, allowed_netlocs):
            if link not in seen_urls:
                queue.append((link, depth + 1))

    return discovered


def wiki_document_from_url(
    url: str,
    allowed_base_url: Optional[str] = None,
) -> Optional[IngestDocument]:
    normalized_url = _normalize_wiki_url(url)
    if normalized_url is None:
        print(f"Skipping invalid wiki url: {url}")
        return None

    allowed_netlocs = _allowed_netlocs_from_base_url(allowed_base_url)
    if allowed_netlocs and not _is_allowed_wiki_url(normalized_url, allowed_netlocs):
        print(f"Skipping disallowed wiki url: {normalized_url}")
        return None

    print(f"Scraping: {normalized_url}")
    html = _fetch_wiki_html(normalized_url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find(id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else None

    content_div = soup.find(id="mw-content-text")
    if not content_div:
        print("No page content found.")
        return None

    for element in content_div.select("script, style, .mw-editsection, #toc"):
        element.decompose()

    raw_text = content_div.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    title, text = title, clean_text
    if not text:
        print(f"Skipping wiki url with no extracted text: {normalized_url}")
        return None

    return IngestDocument(
        content_type="wiki_page",
        canonical_uri=normalized_url,
        title=title,
        external_id=None,
        language="en",
        text=text,
        metadata={},
    )
