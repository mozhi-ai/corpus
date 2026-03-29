"""Tests for deduplication."""

from __future__ import annotations

from pathlib import Path

from mozhi.models import Document
from mozhi.processing.dedup import ExactDedup, NearDedup


def _make_docs(*texts: str) -> list[Document]:
    return [Document(text=t, source="test") for t in texts]


class TestExactDedup:
    def test_removes_exact_duplicates(self) -> None:
        dedup = ExactDedup()
        docs = _make_docs("hello world", "hello world", "unique text")
        result = list(dedup(docs))
        assert len(result) == 2
        assert result[0].text == "hello world"
        assert result[1].text == "unique text"

    def test_whitespace_normalized_dedup(self) -> None:
        """content_hash normalizes whitespace, so these should dedup."""
        dedup = ExactDedup()
        docs = _make_docs("hello  world", "hello world")
        result = list(dedup(docs))
        assert len(result) == 1

    def test_no_duplicates(self) -> None:
        dedup = ExactDedup()
        docs = _make_docs("one", "two", "three")
        result = list(dedup(docs))
        assert len(result) == 3

    def test_all_duplicates(self) -> None:
        dedup = ExactDedup()
        docs = _make_docs("same", "same", "same")
        result = list(dedup(docs))
        assert len(result) == 1

    def test_empty_input(self) -> None:
        dedup = ExactDedup()
        result = list(dedup([]))
        assert result == []

    def test_save_and_load(self, tmp_path: Path) -> None:
        dedup = ExactDedup()
        docs = _make_docs("hello", "world")
        list(dedup(docs))  # populate the hash set

        path = tmp_path / "hashes.txt"
        dedup.save(path)
        assert path.exists()

        loaded = ExactDedup.load(path)
        # "hello" should already be seen
        result = list(loaded(_make_docs("hello", "new text")))
        assert len(result) == 1
        assert result[0].text == "new text"

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        loaded = ExactDedup.load(tmp_path / "nope.txt")
        assert len(loaded._seen) == 0

    def test_name(self) -> None:
        assert ExactDedup().name == "exact_dedup"


class TestNearDedup:
    def test_removes_near_duplicates(self) -> None:
        dedup = NearDedup(threshold=0.5, num_perm=64)
        # These texts are very similar
        docs = _make_docs(
            "தமிழ் ஒரு திராவிட மொழி ஆகும் இது பல நாடுகளில் பேசப்படுகிறது",
            "தமிழ் ஒரு திராவிட மொழி ஆகும் இது பல நாடுகளில் பேசப்படுகின்றது",
            "completely different text that is not related at all to the above",
        )
        result = list(dedup(docs))
        # First and second are near-duplicates, third is different
        assert len(result) == 2

    def test_keeps_unique_documents(self) -> None:
        dedup = NearDedup(threshold=0.8, num_perm=64)
        docs = _make_docs(
            "this is a completely unique document about topic A with lots of content",
            "another entirely different document about topic B with more content here",
        )
        result = list(dedup(docs))
        assert len(result) == 2

    def test_empty_input(self) -> None:
        dedup = NearDedup()
        result = list(dedup([]))
        assert result == []

    def test_name(self) -> None:
        assert NearDedup().name == "near_dedup"
