"""Tamil Unicode normalization."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from mozhi.models import Document

logger = logging.getLogger(__name__)

# Zero-width characters that can be safely removed
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")

# Multiple whitespace
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_tamil_text(text: str) -> str:
    """Apply Tamil-specific Unicode normalization.

    Steps:
    1. NFKC normalization (canonical decomposition + compatibility composition)
    2. Tamil-specific normalization via indic_nlp_library (if available)
    3. Remove zero-width characters
    4. Collapse whitespace
    """
    # Step 1: NFKC normalization — standardizes composed/decomposed forms
    text = unicodedata.normalize("NFKC", text)

    # Step 2: indic_nlp_library Tamil normalizer (handles two-part vowel signs, etc.)
    try:
        from indicnlp.normalize.indic_normalize import TamilNormalizer as IndicTamilNormalizer

        indic_normalizer = IndicTamilNormalizer()
        text = indic_normalizer.normalize(text)
    except ImportError:
        logger.warning("indic_nlp_library not available, skipping Tamil-specific normalization")

    # Step 3: Remove zero-width characters
    text = _ZERO_WIDTH.sub("", text)

    # Step 4: Whitespace normalization
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


class TamilNormalizer:
    """Pipeline step that normalizes Tamil Unicode text."""

    @property
    def name(self) -> str:
        return "tamil_normalizer"

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        count = 0
        for doc in docs:
            doc.text = normalize_tamil_text(doc.text)
            count += 1
            yield doc

        logger.info("Normalizer: processed %d documents", count)
