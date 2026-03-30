"""Tests for the dataset formatter."""

from __future__ import annotations

import json
from pathlib import Path

from mozhi.models import CorpusStats, Document
from mozhi.publishing.formatter import generate_dataset_card, jsonl_to_parquet


class TestJsonlToParquet:
    def test_converts_documents(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "data.jsonl"
        docs = [
            Document(
                text="தமிழ் மொழி",
                source="test",
                url="https://example.com",
                license="CC-0",
                language_score=0.99,
                quality_score=0.85,
                meta={"register": "formal", "domain": "news"},
            ),
            Document(text="வணக்கம்", source="wiki", license="CC-BY-SA"),
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

        parquet_path = tmp_path / "output.parquet"
        count = jsonl_to_parquet(jsonl_path, parquet_path)

        assert count == 2
        assert parquet_path.exists()

        # Verify we can read it back
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        assert table.num_rows == 2
        assert "text" in table.column_names
        assert "source" in table.column_names
        assert "language_score" in table.column_names
        assert "register" in table.column_names
        assert "domain" in table.column_names

    def test_empty_jsonl(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")
        parquet_path = tmp_path / "output.parquet"
        count = jsonl_to_parquet(jsonl_path, parquet_path)
        assert count == 0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "data.jsonl"
        doc = Document(text="test", source="test")
        jsonl_path.write_text(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

        parquet_path = tmp_path / "nested" / "dir" / "output.parquet"
        count = jsonl_to_parquet(jsonl_path, parquet_path)
        assert count == 1
        assert parquet_path.exists()


class TestGenerateDatasetCard:
    def test_contains_stats(self) -> None:
        stats = CorpusStats(
            total_documents=1000,
            total_characters=500_000,
            sources={"wikipedia": 600, "project_madurai": 400},
            licenses={"CC-BY-SA-4.0": 600, "public-domain": 400},
            filtered_count=50,
            dedup_count=20,
        )
        card = generate_dataset_card(stats, "mozhi-ai/tamil-corpus")

        assert "1,000" in card
        assert "500,000" in card
        assert "wikipedia" in card
        assert "project_madurai" in card
        assert "CC-BY-SA-4.0" in card
        assert "mozhi-ai/tamil-corpus" in card

    def test_has_yaml_frontmatter(self) -> None:
        stats = CorpusStats(total_documents=10)
        card = generate_dataset_card(stats, "test/repo")
        assert card.startswith("---")
        assert "language:" in card
        assert "- ta" in card

    def test_has_usage_example(self) -> None:
        stats = CorpusStats(total_documents=10)
        card = generate_dataset_card(stats, "test/repo")
        assert "load_dataset" in card
        assert "test/repo" in card

    def test_has_citation(self) -> None:
        stats = CorpusStats(total_documents=10)
        card = generate_dataset_card(stats, "test/repo")
        assert "@misc" in card
        assert "bibtex" in card

    def test_size_category(self) -> None:
        stats = CorpusStats(total_documents=500)
        card = generate_dataset_card(stats, "test/repo")
        assert "n<1K" in card

        stats2 = CorpusStats(total_documents=50_000)
        card2 = generate_dataset_card(stats2, "test/repo")
        assert "10K<n<100K" in card2
