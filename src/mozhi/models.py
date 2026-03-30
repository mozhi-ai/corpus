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

    # Annotation fields (populated by DocumentAnnotator pipeline step)
    id: str = ""
    domain: str = ""
    register: str = ""
    title: str = ""
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    tamil_char_ratio: float = 0.0
    unique_word_ratio: float = 0.0
    avg_word_length: float = 0.0
    pii_count: int = 0
    has_code_switching: bool = False

    def content_hash(self) -> str:
        """SHA-256 hex digest of the normalized text."""
        normalized = " ".join(self.text.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary suitable for JSONL output."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "url": self.url,
            "license": self.license,
            "date_collected": self.date_collected,
            "domain": self.domain,
            "register": self.register,
            "title": self.title,
            "language_score": self.language_score,
            "quality_score": self.quality_score,
            "tamil_char_ratio": self.tamil_char_ratio,
            "unique_word_ratio": self.unique_word_ratio,
            "avg_word_length": self.avg_word_length,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "sentence_count": self.sentence_count,
            "pii_count": self.pii_count,
            "has_code_switching": self.has_code_switching,
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
            id=data.get("id", ""),
            domain=data.get("domain", ""),
            register=data.get("register", ""),
            title=data.get("title", ""),
            word_count=data.get("word_count", 0),
            char_count=data.get("char_count", 0),
            sentence_count=data.get("sentence_count", 0),
            tamil_char_ratio=data.get("tamil_char_ratio", 0.0),
            unique_word_ratio=data.get("unique_word_ratio", 0.0),
            avg_word_length=data.get("avg_word_length", 0.0),
            pii_count=data.get("pii_count", 0),
            has_code_switching=data.get("has_code_switching", False),
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
    total_words: int = 0
    sources: dict[str, int] = dataclasses.field(default_factory=dict)
    licenses: dict[str, int] = dataclasses.field(default_factory=dict)
    domains: dict[str, int] = dataclasses.field(default_factory=dict)
    registers: dict[str, int] = dataclasses.field(default_factory=dict)
    filtered_count: int = 0
    dedup_count: int = 0

    def update(self, doc: Document) -> None:
        """Update stats with a new document."""
        self.total_documents += 1
        self.total_characters += len(doc.text)
        self.total_words += doc.word_count or len(doc.text.split())
        self.sources[doc.source] = self.sources.get(doc.source, 0) + 1
        if doc.license:
            self.licenses[doc.license] = self.licenses.get(doc.license, 0) + 1
        if doc.domain:
            self.domains[doc.domain] = self.domains.get(doc.domain, 0) + 1
        if doc.register:
            self.registers[doc.register] = self.registers.get(doc.register, 0) + 1

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Documents: {self.total_documents:,}",
            f"Characters: {self.total_characters:,}",
            f"Words: {self.total_words:,}",
            f"Filtered: {self.filtered_count:,}",
            f"Deduplicated: {self.dedup_count:,}",
            "Sources:",
        ]
        for source, count in sorted(self.sources.items()):
            lines.append(f"  {source}: {count:,}")
        if self.domains:
            lines.append("Domains:")
            for domain, count in sorted(self.domains.items()):
                lines.append(f"  {domain}: {count:,}")
        if self.registers:
            lines.append("Registers:")
            for register, count in sorted(self.registers.items()):
                lines.append(f"  {register}: {count:,}")
        return "\n".join(lines)
