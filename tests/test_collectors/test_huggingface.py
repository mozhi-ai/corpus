"""Tests for the HuggingFace collector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mozhi.collectors.huggingface import HuggingFaceCollector
from mozhi.config import CollectorConfig


def _make_collector(tmp_path: Path, datasets: list[dict] | None = None) -> HuggingFaceCollector:  # type: ignore[type-arg]
    """Create a HuggingFaceCollector with test config."""
    if datasets is None:
        datasets = [
            {
                "repo": "test/dataset",
                "subset": "ta",
                "split": "train",
                "license": "CC-0",
            }
        ]
    config = CollectorConfig(name="huggingface", params={"datasets": datasets})
    return HuggingFaceCollector(config, tmp_path)


def _mock_dataset(rows: list[dict]) -> MagicMock:  # type: ignore[type-arg]
    """Create a mock streaming dataset that iterates over rows."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(return_value=iter(rows))
    return mock_ds


# Patch at the source module where load_dataset is actually defined,
# since the collector uses `from datasets import load_dataset` inside the method.
PATCH_LOAD = "datasets.load_dataset"


class TestHuggingFaceCollector:
    def test_name(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        assert collector.name == "huggingface"

    @patch(PATCH_LOAD)
    def test_collect_yields_documents(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _mock_dataset([
            {"text": "தமிழ் மொழி"},
            {"text": "வணக்கம் உலகம்"},
        ])

        collector = _make_collector(tmp_path)
        docs = list(collector.collect())

        assert len(docs) == 2
        assert docs[0].text == "தமிழ் மொழி"
        assert docs[0].source == "hf:test/dataset"
        assert docs[0].license == "CC-0"
        assert docs[1].text == "வணக்கம் உலகம்"

    @patch(PATCH_LOAD)
    def test_collect_with_limit(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _mock_dataset([
            {"text": f"doc {i}"} for i in range(100)
        ])

        collector = _make_collector(tmp_path)
        docs = list(collector.collect(limit=5))

        assert len(docs) == 5

    @patch(PATCH_LOAD)
    def test_skips_empty_text(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _mock_dataset([
            {"text": ""},
            {"text": "   "},
            {"text": "valid text"},
            {"text": None},
        ])

        collector = _make_collector(tmp_path)
        docs = list(collector.collect())

        assert len(docs) == 1
        assert docs[0].text == "valid text"

    @patch(PATCH_LOAD)
    def test_multiple_datasets(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.side_effect = [
            _mock_dataset([{"text": "from dataset 1"}]),
            _mock_dataset([{"text": "from dataset 2"}]),
        ]

        datasets = [
            {"repo": "ds/one", "subset": "ta", "split": "train", "license": "CC-0"},
            {"repo": "ds/two", "subset": "ta", "split": "train", "license": "CC-BY"},
        ]
        collector = _make_collector(tmp_path, datasets=datasets)
        docs = list(collector.collect())

        assert len(docs) == 2
        assert docs[0].source == "hf:ds/one"
        assert docs[1].source == "hf:ds/two"

    @patch(PATCH_LOAD)
    def test_limit_across_datasets(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.side_effect = [
            _mock_dataset([{"text": f"ds1 doc {i}"} for i in range(10)]),
            _mock_dataset([{"text": f"ds2 doc {i}"} for i in range(10)]),
        ]

        datasets = [
            {"repo": "ds/one", "subset": "ta", "split": "train", "license": "CC-0"},
            {"repo": "ds/two", "subset": "ta", "split": "train", "license": "CC-BY"},
        ]
        collector = _make_collector(tmp_path, datasets=datasets)
        docs = list(collector.collect(limit=3))

        assert len(docs) == 3

    @patch(PATCH_LOAD)
    def test_handles_load_failure(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.side_effect = Exception("Network error")
        collector = _make_collector(tmp_path)
        docs = list(collector.collect())
        assert len(docs) == 0

    @patch(PATCH_LOAD)
    def test_metadata_preserved(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _mock_dataset([{"text": "test"}])
        collector = _make_collector(tmp_path)
        docs = list(collector.collect())

        assert docs[0].meta["subset"] == "ta"
        assert docs[0].meta["split"] == "train"
        assert docs[0].date_collected  # not empty
