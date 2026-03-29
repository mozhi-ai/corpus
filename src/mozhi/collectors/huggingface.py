"""Collector for Tamil datasets available on HuggingFace Hub."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mozhi.collectors.base import BaseCollector
from mozhi.models import Document

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mozhi.config import CollectorConfig

logger = logging.getLogger(__name__)


class HuggingFaceCollector(BaseCollector):
    """Collects Tamil text from HuggingFace datasets (CC-100, IndicCorpV2, etc.).

    Uses streaming mode to avoid downloading entire datasets to disk.
    """

    @property
    def name(self) -> str:
        return "huggingface"

    def __init__(self, config: CollectorConfig, corpus_dir: Path) -> None:
        super().__init__(config, corpus_dir)
        self.dataset_configs: list[dict[str, Any]] = self.config.params.get("datasets", [])

    def collect(self, limit: int | None = None) -> Iterator[Document]:
        """Yield documents from all configured HuggingFace datasets."""
        total_yielded = 0

        for ds_config in self.dataset_configs:
            if limit is not None and total_yielded >= limit:
                break

            repo = ds_config["repo"]
            subset = ds_config.get("subset")
            split = ds_config.get("split", "train")
            ds_license = ds_config.get("license", "")
            text_field = ds_config.get("text_field", "text")

            per_dataset_limit = None
            if limit is not None:
                per_dataset_limit = limit - total_yielded

            logger.info("Streaming from %s (subset=%s, split=%s)", repo, subset, split)

            for doc in self._stream_dataset(
                repo=repo,
                subset=subset,
                split=split,
                ds_license=ds_license,
                text_field=text_field,
                limit=per_dataset_limit,
            ):
                yield doc
                total_yielded += 1

        logger.info("HuggingFace collection complete: %d documents", total_yielded)

    def _stream_dataset(
        self,
        *,
        repo: str,
        subset: str | None,
        split: str,
        ds_license: str,
        text_field: str,
        limit: int | None,
    ) -> Iterator[Document]:
        """Stream documents from a single HuggingFace dataset."""
        from datasets import load_dataset

        try:
            ds = load_dataset(repo, subset, split=split, streaming=True)
        except Exception:
            logger.exception("Failed to load dataset %s/%s", repo, subset)
            return

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        count = 0

        for row in ds:
            text = row.get(text_field, "")
            if not text or not isinstance(text, str):
                continue

            text = text.strip()
            if not text:
                continue

            yield Document(
                text=text,
                source=f"hf:{repo}",
                url=f"https://huggingface.co/datasets/{repo}",
                license=ds_license,
                date_collected=today,
                meta={"subset": subset, "split": split},
            )

            count += 1
            if count % 10_000 == 0:
                logger.info("  %s: streamed %d documents so far", repo, count)

            if limit is not None and count >= limit:
                break

        logger.info("  %s: yielded %d documents", repo, count)
