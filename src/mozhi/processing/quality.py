"""Quality filtering using heuristic rules."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from mozhi.models import Document

logger = logging.getLogger(__name__)

# Tamil Unicode block: U+0B80 to U+0BFF
_TAMIL_CHAR_PATTERN = re.compile(r"[\u0B80-\u0BFF]")

# Common boilerplate / low-quality indicators
_BOILERPLATE_PATTERNS = [
    re.compile(r"cookie", re.IGNORECASE),
    re.compile(r"terms\s*(of\s*)?use", re.IGNORECASE),
    re.compile(r"privacy\s*policy", re.IGNORECASE),
    re.compile(r"javascript", re.IGNORECASE),
    re.compile(r"subscribe\s*(now)?", re.IGNORECASE),
    re.compile(r"click\s*here", re.IGNORECASE),
    re.compile(r"\{[^}]*\}"),  # CSS/JS-like curly braces
]


def tamil_char_ratio(text: str) -> float:
    """Calculate the ratio of Tamil characters to total non-whitespace characters."""
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return 0.0
    tamil_chars = len(_TAMIL_CHAR_PATTERN.findall(non_ws))
    return tamil_chars / len(non_ws)


def unique_word_ratio(text: str) -> float:
    """Calculate the ratio of unique words to total words."""
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def has_boilerplate(text: str) -> bool:
    """Check if text contains common boilerplate patterns."""
    return any(p.search(text) for p in _BOILERPLATE_PATTERNS)


class QualityFilter:
    """Filters documents based on heuristic quality rules."""

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 100_000,
        tamil_char_ratio_min: float = 0.5,
        unique_word_ratio_min: float = 0.1,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.tamil_char_ratio_min = tamil_char_ratio_min
        self.unique_word_ratio_min = unique_word_ratio_min
        self._counts: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "quality_filter"

    def _reject(self, reason: str) -> bool:
        self._counts[reason] = self._counts.get(reason, 0) + 1
        return True

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        self._counts = {}
        passed = 0

        for doc in docs:
            text = doc.text

            # Length checks
            if len(text) < self.min_length:
                self._reject("too_short")
                continue
            if len(text) > self.max_length:
                self._reject("too_long")
                continue

            # Tamil character ratio
            ratio = tamil_char_ratio(text)
            if ratio < self.tamil_char_ratio_min:
                self._reject("low_tamil_ratio")
                continue

            # Repetition check
            uwr = unique_word_ratio(text)
            if uwr < self.unique_word_ratio_min:
                self._reject("too_repetitive")
                continue

            # Boilerplate check
            if has_boilerplate(text):
                self._reject("boilerplate")
                continue

            # Set quality score based on Tamil ratio and uniqueness
            doc.quality_score = (ratio + uwr) / 2
            passed += 1
            yield doc

        total_filtered = sum(self._counts.values())
        logger.info(
            "Quality filter: passed %d, filtered %d %s",
            passed,
            total_filtered,
            dict(self._counts) if self._counts else "",
        )
