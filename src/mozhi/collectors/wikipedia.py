"""Collector for Tamil Wikipedia articles via the MediaWiki API."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mozhi.collectors.base import BaseCollector
from mozhi.models import Document

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mozhi.config import CollectorConfig

logger = logging.getLogger(__name__)

# Patterns to clean Wikipedia text
_WIKI_CLEANUP_PATTERNS = [
    (re.compile(r"==+\s*(?:References|External links|See also|Notes)\s*==+.*", re.DOTALL), ""),
    (re.compile(r"\[\[Category:.*?\]\]"), ""),
    (re.compile(r"\{\{.*?\}\}", re.DOTALL), ""),
    (re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]"), r"\1"),  # [[link|text]] → text
    (re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL), ""),
    (re.compile(r"<ref[^/]*/?>"), ""),
    (re.compile(r"<!--.*?-->", re.DOTALL), ""),
    (re.compile(r"<[^>]+>"), ""),  # remaining HTML tags
    (re.compile(r"\'{2,3}"), ""),  # bold/italic markers
    (re.compile(r"==+\s*"), "\n"),  # section headers → newline
    (re.compile(r"\s*==+"), "\n"),
    (re.compile(r"\n{3,}"), "\n\n"),  # collapse multiple newlines
]


def _clean_wiki_text(text: str) -> str:
    """Remove wiki markup and return plain text."""
    for pattern, replacement in _WIKI_CLEANUP_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


class WikipediaCollector(BaseCollector):
    """Collects Tamil Wikipedia articles using the wikipedia-api library.

    Supports incremental collection via checkpointing — tracks which articles
    have been collected so we can resume on the next run.
    """

    @property
    def name(self) -> str:
        return "wikipedia"

    def __init__(self, config: CollectorConfig, corpus_dir: Path) -> None:
        super().__init__(config, corpus_dir)
        self.language = self.config.params.get("language", "ta")
        self.batch_size = self.config.params.get("batch_size", 100)
        self.max_articles: int | None = self.config.params.get("max_articles")

    def collect(self, limit: int | None = None) -> Iterator[Document]:
        """Yield documents from Tamil Wikipedia."""
        import wikipediaapi

        effective_limit = limit or self.max_articles
        wiki = wikipediaapi.Wikipedia(
            user_agent="mozhi-ai/0.1 (Tamil corpus project)",
            language=self.language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

        last_title = self._read_checkpoint("last_title")
        if last_title:
            logger.info("Resuming Wikipedia collection from: %s", last_title)

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        count = 0

        for title, text in self._iter_all_articles(wiki, start_from=last_title):
            cleaned = _clean_wiki_text(text)

            if len(cleaned) < 50:
                continue

            yield Document(
                text=cleaned,
                source="wikipedia",
                url=f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                license="CC-BY-SA-4.0",
                date_collected=today,
                meta={"title": title, "language": self.language},
            )

            count += 1
            if count % 100 == 0:
                logger.info("Wikipedia: collected %d articles", count)
                self._write_checkpoint("last_title", title)

            if effective_limit is not None and count >= effective_limit:
                break

        # Final checkpoint
        if count > 0:
            self._write_checkpoint("last_title", title)  # type: ignore[possibly-undefined]

        logger.info("Wikipedia collection complete: %d articles", count)

    def _iter_all_articles(
        self,
        wiki: object,
        start_from: str | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Iterate through all Wikipedia articles using the allpages API.

        Yields (title, text) tuples.
        """
        import requests

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "mozhi-ai/0.1 (Tamil corpus project; https://github.com/mozhi-ai)",
            }
        )
        api_url = f"https://{self.language}.wikipedia.org/w/api.php"

        params: dict[str, str | int] = {
            "action": "query",
            "list": "allpages",
            "aplimit": self.batch_size,
            "apfilterredir": "nonredirects",
            "format": "json",
        }

        if start_from:
            params["apcontinue"] = start_from

        while True:
            resp = session.get(api_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                break

            for page_info in pages:
                title = page_info["title"]
                page = wiki.page(title)
                if page.exists():
                    text = page.text
                    if text:
                        yield title, text

            # Check for continuation
            continuation = data.get("continue", {})
            if "apcontinue" not in continuation:
                break
            params["apcontinue"] = continuation["apcontinue"]
