"""Tests for document annotation."""

from __future__ import annotations

from mozhi.models import Document
from mozhi.processing.annotator import (
    DocumentAnnotator,
    compute_tamil_char_ratio,
    compute_unique_word_ratio,
    count_sentences,
    detect_code_switching,
    detect_pii,
    infer_domain,
    infer_register,
)


class TestComputeTamilCharRatio:
    def test_pure_tamil(self) -> None:
        assert compute_tamil_char_ratio("தமிழ்மொழி") > 0.9

    def test_pure_english(self) -> None:
        assert compute_tamil_char_ratio("english text") == 0.0

    def test_mixed(self) -> None:
        ratio = compute_tamil_char_ratio("தமிழ் and English")
        assert 0.0 < ratio < 1.0

    def test_empty(self) -> None:
        assert compute_tamil_char_ratio("") == 0.0


class TestComputeUniqueWordRatio:
    def test_all_unique(self) -> None:
        assert compute_unique_word_ratio("one two three") == 1.0

    def test_all_same(self) -> None:
        assert compute_unique_word_ratio("same same same") < 0.5

    def test_empty(self) -> None:
        assert compute_unique_word_ratio("") == 0.0


class TestCountSentences:
    def test_multiple_sentences(self) -> None:
        assert count_sentences("First sentence. Second sentence. Third.") == 3

    def test_question_marks(self) -> None:
        assert count_sentences("What? Why? How?") == 3

    def test_empty(self) -> None:
        assert count_sentences("") == 0

    def test_no_punctuation(self) -> None:
        assert count_sentences("no punctuation here") == 1


class TestDetectPii:
    def test_email(self) -> None:
        assert detect_pii("contact us at test@example.com for info") == 1

    def test_multiple_emails(self) -> None:
        assert detect_pii("a@b.com and c@d.com") == 2

    def test_phone(self) -> None:
        assert detect_pii("call +91 9876543210 now") >= 1

    def test_ip_address(self) -> None:
        assert detect_pii("server at 192.168.1.1") == 1

    def test_no_pii(self) -> None:
        assert detect_pii("தமிழ் மொழி ஒரு அழகான மொழி") == 0


class TestDetectCodeSwitching:
    def test_pure_tamil(self) -> None:
        assert not detect_code_switching("தமிழ் ஒரு திராவிட மொழி ஆகும்")

    def test_tanglish(self) -> None:
        assert detect_code_switching("இது very nice ah, super project da")

    def test_mostly_english(self) -> None:
        assert detect_code_switching("This is English text with some words")

    def test_empty(self) -> None:
        assert not detect_code_switching("")


class TestInferDomain:
    def test_indiccorp(self) -> None:
        assert infer_domain("hf:ai4bharat/IndicCorpV2") == "news"

    def test_c4(self) -> None:
        assert infer_domain("hf:allenai/c4") == "web"

    def test_wikipedia(self) -> None:
        assert infer_domain("wikipedia") == "encyclopedia"

    def test_madurai(self) -> None:
        assert infer_domain("project_madurai") == "classical_literature"

    def test_unknown(self) -> None:
        assert infer_domain("something_new") == "other"


class TestInferRegister:
    def test_indiccorp(self) -> None:
        assert infer_register("hf:ai4bharat/IndicCorpV2") == "formal"

    def test_c4(self) -> None:
        assert infer_register("hf:allenai/c4") == "colloquial"

    def test_wikipedia(self) -> None:
        assert infer_register("wikipedia") == "formal"

    def test_madurai(self) -> None:
        assert infer_register("project_madurai") == "classical"


class TestDocumentAnnotator:
    def test_sets_id(self) -> None:
        step = DocumentAnnotator()
        docs = [Document(text="தமிழ் மொழி", source="test")]
        result = list(step(docs))
        assert result[0].id != ""
        assert len(result[0].id) == 64  # SHA-256 hex

    def test_sets_text_stats(self) -> None:
        step = DocumentAnnotator()
        docs = [Document(text="word1 word2 word3. sentence two.", source="test")]
        result = list(step(docs))
        assert result[0].word_count == 5
        assert result[0].char_count == len("word1 word2 word3. sentence two.")
        assert result[0].sentence_count == 2
        assert result[0].avg_word_length > 0

    def test_sets_domain_and_register(self) -> None:
        step = DocumentAnnotator()
        docs = [Document(text="test", source="wikipedia")]
        result = list(step(docs))
        assert result[0].domain == "encyclopedia"
        assert result[0].register == "formal"

    def test_preserves_existing_domain(self) -> None:
        step = DocumentAnnotator()
        docs = [Document(text="test", source="test", domain="custom")]
        result = list(step(docs))
        assert result[0].domain == "custom"

    def test_preserves_register_from_meta(self) -> None:
        step = DocumentAnnotator()
        docs = [Document(text="test", source="test", meta={"register": "classical"})]
        result = list(step(docs))
        assert result[0].register == "classical"

    def test_extracts_title_from_meta(self) -> None:
        step = DocumentAnnotator()
        docs = [Document(text="test", source="test", meta={"title": "My Title"})]
        result = list(step(docs))
        assert result[0].title == "My Title"

    def test_detects_pii(self) -> None:
        step = DocumentAnnotator()
        docs = [Document(text="email me at test@example.com please", source="test")]
        result = list(step(docs))
        assert result[0].pii_count >= 1

    def test_detects_code_switching(self) -> None:
        step = DocumentAnnotator()
        docs = [Document(text="This is all English text here", source="test")]
        result = list(step(docs))
        assert result[0].has_code_switching is True

    def test_name(self) -> None:
        assert DocumentAnnotator().name == "annotator"
