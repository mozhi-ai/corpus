"""Deduplication: exact (SHA-256) and near-duplicate (MinHash LSH)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from mozhi.models import Document

logger = logging.getLogger(__name__)


class ExactDedup:
    """Removes exact duplicate documents using SHA-256 content hashing.

    Uses an in-memory set of hashes. For persistent cross-run dedup,
    the set can be saved/loaded from disk.
    """

    def __init__(self, seen_hashes: set[str] | None = None) -> None:
        self._seen: set[str] = seen_hashes or set()
        self._dedup_count = 0

    @property
    def name(self) -> str:
        return "exact_dedup"

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        self._dedup_count = 0

        for doc in docs:
            h = doc.content_hash()
            if h in self._seen:
                self._dedup_count += 1
                continue
            self._seen.add(h)
            yield doc

        logger.info(
            "Exact dedup: removed %d duplicates, %d unique hashes tracked",
            self._dedup_count,
            len(self._seen),
        )

    def save(self, path: Path) -> None:
        """Save the hash set to a file for cross-run dedup."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for h in sorted(self._seen):
                f.write(h + "\n")
        logger.info("Saved %d hashes to %s", len(self._seen), path)

    @classmethod
    def load(cls, path: Path) -> ExactDedup:
        """Load a previously saved hash set."""
        if not path.exists():
            return cls()
        with open(path) as f:
            hashes = {line.strip() for line in f if line.strip()}
        logger.info("Loaded %d hashes from %s", len(hashes), path)
        return cls(seen_hashes=hashes)


class NearDedup:
    """Removes near-duplicate documents using MinHash LSH.

    Uses the datasketch library for efficient approximate duplicate detection.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_size: int = 5,
    ) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._dedup_count = 0

    @property
    def name(self) -> str:
        return "near_dedup"

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        from datasketch import MinHashLSH

        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._dedup_count = 0
        doc_id = 0

        for doc in docs:
            mh = self._minhash(doc.text)
            key = f"doc_{doc_id}"

            # Check if similar document already exists
            result = lsh.query(mh)
            if result:
                self._dedup_count += 1
                doc_id += 1
                continue

            import contextlib

            with contextlib.suppress(ValueError):
                lsh.insert(key, mh)

            doc_id += 1
            yield doc

        logger.info(
            "Near dedup: removed %d near-duplicates (threshold=%.2f)",
            self._dedup_count,
            self.threshold,
        )

    def _minhash(self, text: str) -> object:
        """Create a MinHash signature from text using character shingles."""
        from datasketch import MinHash

        mh = MinHash(num_perm=self.num_perm)
        # Use character-level shingles for Tamil (word boundaries are complex)
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i : i + self.shingle_size]
            mh.update(shingle.encode("utf-8"))
        return mh
