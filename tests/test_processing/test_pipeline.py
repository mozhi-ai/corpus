"""Tests for the pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from mozhi.models import Document
from mozhi.processing.pipeline import Pipeline, PipelineStep


class PassthroughStep:
    """A step that passes all documents through unchanged."""

    @property
    def name(self) -> str:
        return "passthrough"

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        yield from docs


class DropShortStep:
    """A step that drops documents shorter than 10 chars."""

    @property
    def name(self) -> str:
        return "drop_short"

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        for doc in docs:
            if len(doc.text) >= 10:
                yield doc


class UpperCaseStep:
    """A step that uppercases all text."""

    @property
    def name(self) -> str:
        return "uppercase"

    def __call__(self, docs: Iterable[Document]) -> Iterator[Document]:
        for doc in docs:
            doc.text = doc.text.upper()
            yield doc


def _make_docs(*texts: str) -> list[Document]:
    return [Document(text=t, source="test") for t in texts]


class TestPipeline:
    def test_empty_pipeline(self) -> None:
        pipeline = Pipeline(steps=[])
        docs = _make_docs("hello", "world")
        result = list(pipeline.run(docs))
        assert len(result) == 2

    def test_single_step(self) -> None:
        pipeline = Pipeline(steps=[PassthroughStep()])
        docs = _make_docs("hello", "world")
        result = list(pipeline.run(docs))
        assert len(result) == 2

    def test_filtering_step(self) -> None:
        pipeline = Pipeline(steps=[DropShortStep()])
        docs = _make_docs("short", "this is long enough")
        result = list(pipeline.run(docs))
        assert len(result) == 1
        assert result[0].text == "this is long enough"

    def test_transform_step(self) -> None:
        pipeline = Pipeline(steps=[UpperCaseStep()])
        docs = _make_docs("hello")
        result = list(pipeline.run(docs))
        assert result[0].text == "HELLO"

    def test_chained_steps(self) -> None:
        pipeline = Pipeline(steps=[DropShortStep(), UpperCaseStep()])
        docs = _make_docs("hi", "hello world")
        result = list(pipeline.run(docs))
        assert len(result) == 1
        assert result[0].text == "HELLO WORLD"

    def test_steps_are_pipeline_step_protocol(self) -> None:
        assert isinstance(PassthroughStep(), PipelineStep)
        assert isinstance(DropShortStep(), PipelineStep)
