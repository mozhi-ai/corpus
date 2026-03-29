"""I/O utilities for JSONL files and path management."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

from mozhi.models import Document


def ensure_dirs(corpus_dir: Path) -> None:
    """Create the corpus directory structure if it doesn't exist."""
    for subdir in ("raw", "processed", "published", "dedup_index"):
        (corpus_dir / subdir).mkdir(parents=True, exist_ok=True)


def write_jsonl(docs: Iterable[Document], path: Path) -> int:
    """Write documents to a JSONL file. Returns the number of documents written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Document]:
    """Read documents from a JSONL file as a lazy iterator."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Document.from_json(line)
