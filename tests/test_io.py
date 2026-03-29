"""Tests for mozhi.utils.io."""

from pathlib import Path

from mozhi.models import Document
from mozhi.utils.io import ensure_dirs, read_jsonl, write_jsonl


def test_write_and_read_jsonl(tmp_path: Path) -> None:
    docs = [
        Document(text="வணக்கம்", source="test"),
        Document(text="நன்றி", source="test", url="https://example.com"),
    ]
    path = tmp_path / "output.jsonl"
    count = write_jsonl(docs, path)
    assert count == 2
    assert path.exists()

    restored = list(read_jsonl(path))
    assert len(restored) == 2
    assert restored[0].text == "வணக்கம்"
    assert restored[1].url == "https://example.com"


def test_write_jsonl_appends(tmp_path: Path) -> None:
    path = tmp_path / "output.jsonl"
    write_jsonl([Document(text="first", source="a")], path)
    write_jsonl([Document(text="second", source="b")], path)
    docs = list(read_jsonl(path))
    assert len(docs) == 2
    assert docs[0].text == "first"
    assert docs[1].text == "second"


def test_write_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "output.jsonl"
    write_jsonl([Document(text="test", source="a")], path)
    assert path.exists()


def test_ensure_dirs(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    ensure_dirs(corpus_dir)
    assert (corpus_dir / "raw").is_dir()
    assert (corpus_dir / "processed").is_dir()
    assert (corpus_dir / "published").is_dir()
    assert (corpus_dir / "dedup_index").is_dir()


def test_ensure_dirs_idempotent(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    ensure_dirs(corpus_dir)
    ensure_dirs(corpus_dir)  # should not raise
    assert (corpus_dir / "raw").is_dir()
