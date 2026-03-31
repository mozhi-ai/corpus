"""Document annotation — computes metadata fields for each document."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from mozhi.models import Document

logger = logging.getLogger(__name__)

# Tamil Unicode block
_TAMIL_CHAR = re.compile(r"[\u0B80-\u0BFF]")

# English ASCII letters
_ENGLISH_CHAR = re.compile(r"[a-zA-Z]")

# Tamil sentence-ending punctuation
_SENTENCE_END = re.compile(r"[.!?।]")

# PII patterns
_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE = re.compile(r"(?:\+91[\s-]?)?(?:\d[\s-]?){10,13}")
_IP_ADDR = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# Source → domain mapping
_SOURCE_DOMAIN_MAP: dict[str, str] = {
    "hf:ai4bharat/IndicCorpV2": "news",
    "hf:allenai/c4": "web",
    "hf:AnanthZeke/oscar_tamil_clean": "web",
    "hf:Hemanth-thunder/tamil-madlad-400": "web",
    "hf:livinNector/tamil_news_dataset": "news",
    "wikipedia": "encyclopedia",
    "project_madurai": "classical_literature",
}

# Source → register mapping (defaults, can be overridden)
_SOURCE_REGISTER_MAP: dict[str, str] = {
    "hf:ai4bharat/IndicCorpV2": "formal",
    "hf:allenai/c4": "colloquial",
    "hf:AnanthZeke/oscar_tamil_clean": "colloquial",
    "hf:Hemanth-thunder/tamil-madlad-400": "colloquial",
    "hf:livinNector/tamil_news_dataset": "formal",
    "wikipedia": "formal",
    "project_madurai": "classical",
}


def compute_tamil_char_ratio(text: str) -> float:
    """Ratio of Tamil characters to total non-whitespace characters."""
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return 0.0
    return len(_TAMIL_CHAR.findall(non_ws)) / len(non_ws)


def compute_unique_word_ratio(text: str) -> float:
    """Ratio of unique words to total words."""
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def count_sentences(text: str) -> int:
    """Count sentences using Tamil/common sentence-ending punctuation."""
    ends = _SENTENCE_END.findall(text)
    return max(len(ends), 1) if text.strip() else 0


def detect_pii(text: str) -> int:
    """Count PII instances (emails, phone numbers, IP addresses)."""
    return len(_EMAIL.findall(text)) + len(_PHONE.findall(text)) + len(_IP_ADDR.findall(text))


def detect_code_switching(text: str) -> bool:
    """Detect if text contains significant English (code-switching / Tanglish)."""
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return False
    english_ratio = len(_ENGLISH_CHAR.findall(non_ws)) / len(non_ws)
    # If >15% English characters, likely code-switched
    return english_ratio > 0.15


def infer_domain(source: str) -> str:
    """Infer content domain from source identifier."""
    for prefix, domain in _SOURCE_DOMAIN_MAP.items():
        if source.startswith(prefix) or source == prefix:
            return domain
    return "other"


def infer_register(source: str) -> str:
    """Infer language register from source identifier."""
    for prefix, register in _SOURCE_REGISTER_MAP.items():
        if source.startswith(prefix) or source == prefix:
            return register
    return "unknown"


class DocumentAnnotator:
    """Pipeline step that computes all annotation fields on each document."""

    @property
    def name(self) -> str:
        return "annotator"

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        count = 0

        for doc in docs:
            # ID
            doc.id = doc.content_hash()

            # Domain and register (use existing values if set, otherwise infer)
            if not doc.domain:
                doc.domain = doc.meta.get("domain", "") or infer_domain(doc.source)
            if not doc.register:
                doc.register = doc.meta.get("register", "") or infer_register(doc.source)

            # Title from meta
            if not doc.title:
                doc.title = doc.meta.get("title", "")

            # Text statistics
            words = doc.text.split()
            doc.char_count = len(doc.text)
            doc.word_count = len(words)
            doc.sentence_count = count_sentences(doc.text)
            doc.avg_word_length = doc.char_count / doc.word_count if doc.word_count else 0.0

            # Quality signals
            doc.tamil_char_ratio = compute_tamil_char_ratio(doc.text)
            doc.unique_word_ratio = compute_unique_word_ratio(doc.text)

            # Safety signals
            doc.pii_count = detect_pii(doc.text)
            doc.has_code_switching = detect_code_switching(doc.text)

            count += 1
            yield doc

        logger.info("Annotator: enriched %d documents", count)
