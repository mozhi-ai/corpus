"""Collector for Project Madurai classical Tamil literature."""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mozhi.collectors.base import BaseCollector
from mozhi.models import Document

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mozhi.config import CollectorConfig

logger = logging.getLogger(__name__)

# Project Madurai has ~395 texts numbered pmuni0001 to pmuni0395
BASE_URL = "https://www.projectmadurai.org/pm_etexts/utf8"

# Lines matching these patterns are PM boilerplate (English metadata/credits)
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"project\s*madurai", re.IGNORECASE),
    re.compile(r"Dr\.?\s*K\.?\s*Kalyanasundaram", re.IGNORECASE),
    re.compile(r"Lausanne", re.IGNORECASE),
    re.compile(r"©|copyright", re.IGNORECASE),
    re.compile(r"This\s+(file|etext|page)", re.IGNORECASE),
    re.compile(r"The Etext", re.IGNORECASE),
    re.compile(r"send your comments", re.IGNORECASE),
    re.compile(r"was (first|last) (put up|revised)", re.IGNORECASE),
    re.compile(r"Webmaster|pmadurai@", re.IGNORECASE),
    re.compile(r"Preparation\s+of", re.IGNORECASE),
    re.compile(r"Etext\s+(Coverage|in)", re.IGNORECASE),
    re.compile(r"^\s*<!DOCTYPE", re.IGNORECASE),
    re.compile(r"in\s+tamil\s+script", re.IGNORECASE),
    re.compile(r"unicode(/UTF-8)?\s+format", re.IGNORECASE),
    re.compile(r"Acknowledgements?", re.IGNORECASE),
    re.compile(r"content.*?converted to Unicode", re.IGNORECASE),
    re.compile(r"electronic texts? of tamil", re.IGNORECASE),
    re.compile(r"distribute them free", re.IGNORECASE),
    re.compile(r"presented in Unicode", re.IGNORECASE),
    re.compile(r"^\s*Works? of\s+\w+\s*:", re.IGNORECASE),
    re.compile(r"Internet\.\s*$", re.IGNORECASE),
]


def _extract_text(html: str) -> str:
    """Extract clean Tamil text from Project Madurai HTML."""
    from html.parser import HTMLParser

    # Step 1: Use HTMLParser to extract text content, skipping script/style
    class TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in ("script", "style", "head"):
                self._skip = True
            elif tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "tr"):
                self.parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in ("script", "style", "head"):
                self._skip = False

        def handle_data(self, data: str) -> None:
            if not self._skip:
                self.parts.append(data)

    parser = TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)

    # Step 2: Remove boilerplate lines (English metadata, credits, etc.)
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if any(p.search(stripped) for p in _BOILERPLATE_LINE_PATTERNS):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Step 3: Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_title(html: str) -> str:
    """Try to extract the title from the HTML."""
    # Try <title> tag first
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        if title:
            return title

    # Try first <h1> or <h2>
    for tag in ("h1", "h2", "h3"):
        match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            if title:
                return title

    return "Unknown"


class MaduraiCollector(BaseCollector):
    """Collects classical Tamil literature from Project Madurai.

    Project Madurai hosts ~395 classical Tamil literary works in UTF-8 HTML
    format at https://www.projectmadurai.org/pm_etexts/utf8/pmuniNNNN.html
    """

    @property
    def name(self) -> str:
        return "madurai"

    def __init__(self, config: CollectorConfig, corpus_dir: Path) -> None:
        super().__init__(config, corpus_dir)
        self.start_id: int = self.config.params.get("start_id", 1)
        self.end_id: int = self.config.params.get("end_id", 395)
        self.delay: float = self.config.params.get("delay_seconds", 2.0)

    def collect(self, limit: int | None = None) -> Iterator[Document]:
        """Yield documents from Project Madurai texts."""
        import requests

        session = requests.Session()
        session.headers.update({
            "User-Agent": "mozhi-ai/0.1 (Tamil corpus project; educational use)",
        })

        # Resume from checkpoint
        last_id_str = self._read_checkpoint("last_id")
        start = int(last_id_str) + 1 if last_id_str else self.start_id

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        count = 0

        for text_id in range(start, self.end_id + 1):
            if limit is not None and count >= limit:
                break

            filename = f"pmuni{text_id:04d}.html"
            url = f"{BASE_URL}/{filename}"

            try:
                resp = session.get(url, timeout=30)
                if resp.status_code == 404:
                    logger.debug("Not found: %s (skipping)", url)
                    continue
                resp.raise_for_status()
                resp.encoding = "utf-8"
            except Exception:
                logger.warning("Failed to fetch %s, skipping", url)
                continue

            html = resp.text
            title = _extract_title(html)
            text = _extract_text(html)

            if len(text) < 100:
                logger.debug("Too short after cleaning: %s (%d chars)", filename, len(text))
                continue

            yield Document(
                text=text,
                source="project_madurai",
                url=url,
                license="public-domain",
                date_collected=today,
                meta={
                    "title": title,
                    "text_id": text_id,
                    "filename": filename,
                    "register": "classical",
                },
            )

            count += 1
            self._write_checkpoint("last_id", str(text_id))

            if count % 10 == 0:
                logger.info("Project Madurai: collected %d texts", count)

            # Be respectful — don't hammer the server
            time.sleep(self.delay)

        logger.info("Project Madurai collection complete: %d texts", count)
