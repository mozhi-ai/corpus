"""Core data models for the mozhi corpus pipeline."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from typing import Any


@dataclasses.dataclass
class Document:
    """A single text document flowing through the corpus pipeline."""

    text: str
    source: str
    url: str | None = None
    license: str = ""
    date_collected: str = dataclasses.field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d")
    )
    language_score: float = 0.0
    quality_score: float = 0.0
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def content_hash(self) -> str:
        """SHA-256 hex digest of the normalized text."""
        normalized = " ".join(self.text.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary suitable for JSONL output."""
        return {
            "text": self.text,
            "source": self.source,
            "url": self.url,
            "license": self.license,
            "date_collected": self.date_collected,
            "language_score": self.language_score,
            "quality_score": self.quality_score,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Deserialize from a dictionary."""
        return cls(
            text=data["text"],
            source=data["source"],
            url=data.get("url"),
            license=data.get("license", ""),
            date_collected=data.get("date_collected", ""),
            language_score=data.get("language_score", 0.0),
            quality_score=data.get("quality_score", 0.0),
            meta=data.get("meta", {}),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Document:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(line))


@dataclasses.dataclass
class CorpusStats:
    """Statistics about a corpus or processing run."""

    total_documents: int = 0
    total_characters: int = 0
    sources: dict[str, int] = dataclasses.field(default_factory=dict)
    licenses: dict[str, int] = dataclasses.field(default_factory=dict)
    filtered_count: int = 0
    dedup_count: int = 0

    @property
    def total_words(self) -> int:
        """Rough word count estimate."""
        return self.total_characters // 5  # approximate for Tamil

    def update(self, doc: Document) -> None:
        """Update stats with a new document."""
        self.total_documents += 1
        self.total_characters += len(doc.text)
        self.sources[doc.source] = self.sources.get(doc.source, 0) + 1
        if doc.license:
            self.licenses[doc.license] = self.licenses.get(doc.license, 0) + 1

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Documents: {self.total_documents:,}",
            f"Characters: {self.total_characters:,}",
            f"Est. words: {self.total_words:,}",
            f"Filtered: {self.filtered_count:,}",
            f"Deduplicated: {self.dedup_count:,}",
            "Sources:",
        ]
        for source, count in sorted(self.sources.items()):
            lines.append(f"  {source}: {count:,}")
        return "\n".join(lines)
