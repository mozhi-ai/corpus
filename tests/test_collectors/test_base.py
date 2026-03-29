"""Tests for the base collector and collector registry."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mozhi.collectors import get_collector
from mozhi.collectors.base import BaseCollector
from mozhi.config import CollectorConfig
from mozhi.models import Document

if TYPE_CHECKING:
    from collections.abc import Iterator


class DummyCollector(BaseCollector):
    """Minimal concrete collector for testing the base class."""

    @property
    def name(self) -> str:
        return "dummy"

    def collect(self, limit: int | None = None) -> Iterator[Document]:
        docs = [
            Document(text="doc one", source="dummy"),
            Document(text="doc two", source="dummy"),
            Document(text="doc three", source="dummy"),
        ]
        for i, doc in enumerate(docs):
            if limit is not None and i >= limit:
                break
            yield doc


class TestBaseCollector:
    def test_creates_raw_dir(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="dummy")
        DummyCollector(config, tmp_path)
        assert (tmp_path / "raw" / "dummy").is_dir()

    def test_collect_yields_documents(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="dummy")
        collector = DummyCollector(config, tmp_path)
        docs = list(collector.collect())
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_collect_with_limit(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="dummy")
        collector = DummyCollector(config, tmp_path)
        docs = list(collector.collect(limit=2))
        assert len(docs) == 2

    def test_save_raw(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="dummy")
        collector = DummyCollector(config, tmp_path)
        docs = [Document(text="hello", source="test")]
        count = collector.save_raw(docs, "test_batch")
        assert count == 1
        assert (tmp_path / "raw" / "dummy" / "test_batch.jsonl").exists()

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="dummy")
        collector = DummyCollector(config, tmp_path)
        assert collector._read_checkpoint("key") is None
        collector._write_checkpoint("key", "value123")
        assert collector._read_checkpoint("key") == "value123"

    def test_checkpoint_update(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="dummy")
        collector = DummyCollector(config, tmp_path)
        collector._write_checkpoint("a", "1")
        collector._write_checkpoint("b", "2")
        assert collector._read_checkpoint("a") == "1"
        assert collector._read_checkpoint("b") == "2"
        collector._write_checkpoint("a", "updated")
        assert collector._read_checkpoint("a") == "updated"


class TestCollectorRegistry:
    def test_get_known_collector(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="huggingface", params={"datasets": []})
        collector = get_collector(config, tmp_path)
        assert collector.name == "huggingface"

    def test_get_wikipedia_collector(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="wikipedia", params={"language": "ta"})
        collector = get_collector(config, tmp_path)
        assert collector.name == "wikipedia"

    def test_unknown_collector_raises(self, tmp_path: Path) -> None:
        config = CollectorConfig(name="nonexistent")
        with pytest.raises(ValueError, match="Unknown collector"):
            get_collector(config, tmp_path)
