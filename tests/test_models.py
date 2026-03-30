"""Tests for mozhi.models."""

from mozhi.models import CorpusStats, Document


class TestDocument:
    def test_create_minimal(self) -> None:
        doc = Document(text="வணக்கம்", source="test")
        assert doc.text == "வணக்கம்"
        assert doc.source == "test"
        assert doc.url is None
        assert doc.language_score == 0.0

    def test_content_hash_deterministic(self) -> None:
        doc1 = Document(text="hello world", source="a")
        doc2 = Document(text="hello world", source="b")
        assert doc1.content_hash() == doc2.content_hash()

    def test_content_hash_normalizes_whitespace(self) -> None:
        doc1 = Document(text="hello  world", source="a")
        doc2 = Document(text="hello world", source="a")
        assert doc1.content_hash() == doc2.content_hash()

    def test_content_hash_differs_for_different_text(self) -> None:
        doc1 = Document(text="hello", source="a")
        doc2 = Document(text="world", source="a")
        assert doc1.content_hash() != doc2.content_hash()

    def test_to_dict_roundtrip(self) -> None:
        doc = Document(
            text="தமிழ்",
            source="test",
            url="https://example.com",
            license="CC-0",
            language_score=0.95,
            quality_score=0.8,
            meta={"key": "value"},
        )
        restored = Document.from_dict(doc.to_dict())
        assert restored.text == doc.text
        assert restored.source == doc.source
        assert restored.url == doc.url
        assert restored.license == doc.license
        assert restored.language_score == doc.language_score
        assert restored.meta == doc.meta

    def test_annotation_fields_roundtrip(self) -> None:
        doc = Document(
            text="test",
            source="test",
            id="abc123",
            domain="news",
            register="formal",
            title="My Title",
            word_count=10,
            char_count=50,
            sentence_count=2,
            tamil_char_ratio=0.9,
            unique_word_ratio=0.8,
            avg_word_length=5.0,
            pii_count=1,
            has_code_switching=True,
        )
        restored = Document.from_dict(doc.to_dict())
        assert restored.id == "abc123"
        assert restored.domain == "news"
        assert restored.register == "formal"
        assert restored.title == "My Title"
        assert restored.word_count == 10
        assert restored.pii_count == 1
        assert restored.has_code_switching is True

    def test_json_roundtrip(self) -> None:
        doc = Document(text="தமிழ் மொழி", source="wiki", url=None)
        json_str = doc.to_json()
        restored = Document.from_json(json_str)
        assert restored.text == doc.text
        assert restored.source == doc.source

    def test_json_handles_unicode(self) -> None:
        doc = Document(text="தமிழ் நாடு 🇮🇳", source="test")
        json_str = doc.to_json()
        assert "தமிழ்" in json_str  # ensure_ascii=False
        restored = Document.from_json(json_str)
        assert restored.text == doc.text


class TestCorpusStats:
    def test_update(self) -> None:
        stats = CorpusStats()
        doc = Document(text="hello world", source="test", license="CC-0")
        stats.update(doc)
        assert stats.total_documents == 1
        assert stats.total_characters == 11
        assert stats.sources == {"test": 1}
        assert stats.licenses == {"CC-0": 1}

    def test_update_multiple_sources(self) -> None:
        stats = CorpusStats()
        stats.update(Document(text="abc", source="a", license="CC-0"))
        stats.update(Document(text="def", source="b", license="CC-BY"))
        stats.update(Document(text="ghi", source="a", license="CC-0"))
        assert stats.sources == {"a": 2, "b": 1}
        assert stats.licenses == {"CC-0": 2, "CC-BY": 1}

    def test_summary_contains_key_info(self) -> None:
        stats = CorpusStats(total_documents=100, total_characters=5000)
        stats.sources = {"wiki": 60, "cc100": 40}
        summary = stats.summary()
        assert "100" in summary
        assert "5,000" in summary
        assert "wiki" in summary
