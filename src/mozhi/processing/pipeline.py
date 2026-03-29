"""Pipeline orchestrator that chains processing steps."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from mozhi.config import PipelineConfig
    from mozhi.models import Document

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineStep(Protocol):
    """Protocol for a single processing step in the pipeline."""

    @property
    def name(self) -> str: ...

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]: ...


class Pipeline:
    """Chains multiple PipelineStep callables into a linear processing pipeline.

    Documents flow as iterators through each step — streaming, not materialized.
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        self.steps = steps

    def run(self, docs: Iterable[Document]) -> Iterator[Document]:
        """Run all steps in sequence, yielding processed documents."""
        current: Iterable[Document] = docs
        for step in self.steps:
            logger.info("Running step: %s", step.name)
            current = step(current)
        yield from current

    @classmethod
    def default(
        cls,
        config: PipelineConfig,
        dedup: PipelineStep | None = None,
    ) -> Pipeline:
        """Build the standard pipeline: language → normalize → dedup → quality."""
        from mozhi.processing.dedup import ExactDedup
        from mozhi.processing.language import LanguageFilter
        from mozhi.processing.normalizer import TamilNormalizer
        from mozhi.processing.quality import QualityFilter

        steps: list[PipelineStep] = [
            LanguageFilter(min_score=config.min_language_score),
            TamilNormalizer(),
            dedup or ExactDedup(),
            QualityFilter(
                min_length=config.min_text_length,
                max_length=config.max_text_length,
                tamil_char_ratio_min=config.tamil_char_ratio_min,
            ),
        ]
        return cls(steps)
