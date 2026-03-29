"""Collector registry — maps collector names to their implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mozhi.collectors.base import BaseCollector
    from mozhi.config import CollectorConfig

_REGISTRY: dict[str, type[BaseCollector]] = {}


def _ensure_registry() -> None:
    """Lazily populate the registry to avoid import-time side effects."""
    if _REGISTRY:
        return

    from mozhi.collectors.huggingface import HuggingFaceCollector
    from mozhi.collectors.madurai import MaduraiCollector
    from mozhi.collectors.wikipedia import WikipediaCollector

    _REGISTRY["huggingface"] = HuggingFaceCollector
    _REGISTRY["wikipedia"] = WikipediaCollector
    _REGISTRY["madurai"] = MaduraiCollector


def get_collector(config: CollectorConfig, corpus_dir: Path) -> BaseCollector:
    """Instantiate a collector by name from config."""
    _ensure_registry()
    cls = _REGISTRY.get(config.name)
    if cls is None:
        msg = f"Unknown collector: {config.name!r}. Available: {list(_REGISTRY.keys())}"
        raise ValueError(msg)
    return cls(config, corpus_dir)
