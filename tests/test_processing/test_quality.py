"""Tests for quality filtering."""

from __future__ import annotations

from mozhi.models import Document
from mozhi.processing.quality import (
    QualityFilter,
    has_boilerplate,
    tamil_char_ratio,
    unique_word_ratio,
)


class TestTamilCharRatio:
    def test_pure_tamil(self) -> None:
        text = "தமிழ்மொழி"
        ratio = tamil_char_ratio(text)
        assert ratio > 0.9

    def test_pure_english(self) -> None:
        text = "english only text"
        ratio = tamil_char_ratio(text)
        assert ratio == 0.0

    def test_mixed(self) -> None:
        text = "தமிழ் and English"
        ratio = tamil_char_ratio(text)
        assert 0.0 < ratio < 1.0

    def test_empty(self) -> None:
        assert tamil_char_ratio("") == 0.0

    def test_whitespace_only(self) -> None:
        assert tamil_char_ratio("   ") == 0.0


class TestUniqueWordRatio:
    def test_all_unique(self) -> None:
        assert unique_word_ratio("one two three four") == 1.0

    def test_all_same(self) -> None:
        assert unique_word_ratio("same same same same") == 0.25

    def test_empty(self) -> None:
        assert unique_word_ratio("") == 0.0


class TestHasBoilerplate:
    def test_cookie_text(self) -> None:
        assert has_boilerplate("We use cookies to improve your experience")

    def test_terms_of_use(self) -> None:
        assert has_boilerplate("Please read our Terms of Use")

    def test_privacy_policy(self) -> None:
        assert has_boilerplate("Read our Privacy Policy")

    def test_clean_tamil(self) -> None:
        assert not has_boilerplate("தமிழ் ஒரு திராவிட மொழி ஆகும்")

    def test_curly_braces(self) -> None:
        assert has_boilerplate("body { color: red; }")


class TestQualityFilter:
    def test_passes_good_tamil(self) -> None:
        qf = QualityFilter(min_length=10, tamil_char_ratio_min=0.3)
        docs = [Document(text="தமிழ் ஒரு திராவிட மொழி ஆகும் இது பல நாடுகளில்", source="test")]
        result = list(qf(docs))
        assert len(result) == 1
        assert result[0].quality_score > 0

    def test_filters_too_short(self) -> None:
        qf = QualityFilter(min_length=50)
        docs = [Document(text="குறுகிய", source="test")]
        result = list(qf(docs))
        assert len(result) == 0

    def test_filters_too_long(self) -> None:
        qf = QualityFilter(max_length=100)
        docs = [Document(text="அ" * 200, source="test")]
        result = list(qf(docs))
        assert len(result) == 0

    def test_filters_low_tamil_ratio(self) -> None:
        qf = QualityFilter(min_length=10, tamil_char_ratio_min=0.5)
        docs = [Document(text="This is all English text nothing Tamil here at all", source="test")]
        result = list(qf(docs))
        assert len(result) == 0

    def test_filters_repetitive(self) -> None:
        qf = QualityFilter(min_length=10, tamil_char_ratio_min=0.0, unique_word_ratio_min=0.1)
        docs = [Document(text="spam " * 100, source="test")]
        result = list(qf(docs))
        assert len(result) == 0

    def test_filters_boilerplate(self) -> None:
        qf = QualityFilter(min_length=10, tamil_char_ratio_min=0.0)
        text = "We use cookies to improve your experience on this site"
        docs = [Document(text=text, source="test")]
        result = list(qf(docs))
        assert len(result) == 0

    def test_name(self) -> None:
        assert QualityFilter().name == "quality_filter"
