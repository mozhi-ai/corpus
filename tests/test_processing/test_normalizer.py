"""Tests for Tamil Unicode normalization."""

from __future__ import annotations

from mozhi.models import Document
from mozhi.processing.normalizer import TamilNormalizer, normalize_tamil_text


class TestNormalizeTamilText:
    def test_nfkc_normalization(self) -> None:
        # NFKC should normalize compatibility characters
        text = "\uff21\uff22\uff23"  # fullwidth ABC
        result = normalize_tamil_text(text)
        assert result == "ABC"

    def test_removes_zero_width_chars(self) -> None:
        text = "தமிழ்\u200b மொழி\u200c"
        result = normalize_tamil_text(text)
        assert "\u200b" not in result
        assert "\u200c" not in result
        assert "தமிழ்" in result

    def test_removes_bom(self) -> None:
        text = "\ufeffதமிழ் மொழி"
        result = normalize_tamil_text(text)
        assert not result.startswith("\ufeff")

    def test_collapses_whitespace(self) -> None:
        text = "தமிழ்   மொழி\t\tஆகும்"
        result = normalize_tamil_text(text)
        assert "   " not in result
        assert "\t" not in result

    def test_collapses_newlines(self) -> None:
        text = "line one\n\n\n\n\nline two"
        result = normalize_tamil_text(text)
        assert "\n\n\n" not in result
        assert "line one" in result
        assert "line two" in result

    def test_strips_leading_trailing(self) -> None:
        text = "  \n  தமிழ் மொழி  \n  "
        result = normalize_tamil_text(text)
        assert result == "தமிழ் மொழி"

    def test_preserves_tamil_text(self) -> None:
        text = "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு"
        result = normalize_tamil_text(text)
        assert result == text

    def test_empty_string(self) -> None:
        assert normalize_tamil_text("") == ""


class TestTamilNormalizerStep:
    def test_normalizes_documents(self) -> None:
        step = TamilNormalizer()
        docs = [
            Document(text="தமிழ்\u200b மொழி", source="test"),
            Document(text="  hello  world  ", source="test"),
        ]
        result = list(step(docs))
        assert len(result) == 2
        assert "\u200b" not in result[0].text
        assert result[1].text == "hello world"

    def test_name(self) -> None:
        assert TamilNormalizer().name == "tamil_normalizer"
