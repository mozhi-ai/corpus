"""Base collector interface for all data sources."""

from __future__ import annotations

import abc
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from mozhi.config import CollectorConfig
    from mozhi.models import Document

logger = logging.getLogger(__name__)


class BaseCollector(abc.ABC):
    """Abstract base class for all corpus collectors.

    Each collector knows how to fetch documents from a single source type
    and yield them as an iterator of Document objects.
    """

    def __init__(self, config: CollectorConfig, corpus_dir: Path) -> None:
        self.config = config
        self.corpus_dir = corpus_dir
        self.raw_dir = corpus_dir / "raw" / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique name for this collector (used in directory names and logs)."""

    @abc.abstractmethod
    def collect(self, limit: int | None = None) -> Iterator[Document]:
        """Yield documents from the source.

        Args:
            limit: Maximum number of documents to yield. None means all available.
        """

    def save_raw(self, docs: Iterable[Document], filename: str) -> int:
        """Write documents to corpus/raw/{name}/{filename}.jsonl.

        Returns the number of documents written.
        """
        path = self.raw_dir / f"{filename}.jsonl"
        count = 0
        with open(path, "a", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        logger.info("Saved %d documents to %s", count, path)
        return count

    def _read_checkpoint(self, key: str) -> str | None:
        """Read a checkpoint value for incremental collection."""
        checkpoint_path = self.raw_dir / ".checkpoint"
        if not checkpoint_path.exists():
            return None
        data = json.loads(checkpoint_path.read_text())
        return data.get(key)

    def _write_checkpoint(self, key: str, value: str) -> None:
        """Write a checkpoint value for incremental collection."""
        checkpoint_path = self.raw_dir / ".checkpoint"
        data: dict[str, str] = {}
        if checkpoint_path.exists():
            data = json.loads(checkpoint_path.read_text())
        data[key] = value
        checkpoint_path.write_text(json.dumps(data))
