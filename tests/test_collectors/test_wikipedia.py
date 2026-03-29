"""Tests for the Wikipedia collector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mozhi.collectors.wikipedia import WikipediaCollector, _clean_wiki_text
from mozhi.config import CollectorConfig


def _make_collector(tmp_path: Path, **params: object) -> WikipediaCollector:
    """Create a WikipediaCollector with test config."""
    default_params: dict[str, object] = {"language": "ta", "batch_size": 10}
    default_params.update(params)
    config = CollectorConfig(name="wikipedia", params=default_params)
    return WikipediaCollector(config, tmp_path)


class TestCleanWikiText:
    def test_removes_html_tags(self) -> None:
        text = "<p>தமிழ் <b>மொழி</b></p>"
        assert _clean_wiki_text(text) == "தமிழ் மொழி"

    def test_removes_wiki_links_keeps_text(self) -> None:
        text = "[[திராவிட மொழிகள்|திராவிட]] மொழி"
        assert _clean_wiki_text(text) == "திராவிட மொழி"

    def test_removes_simple_wiki_links(self) -> None:
        text = "[[தமிழ்]] மொழி"
        assert _clean_wiki_text(text) == "தமிழ் மொழி"

    def test_removes_templates(self) -> None:
        text = "தமிழ் {{cite|source=foo}} மொழி"
        assert _clean_wiki_text(text) == "தமிழ்  மொழி"

    def test_removes_references(self) -> None:
        text = 'தமிழ்<ref name="foo">citation text</ref> மொழி'
        assert _clean_wiki_text(text) == "தமிழ் மொழி"

    def test_removes_self_closing_refs(self) -> None:
        text = 'தமிழ்<ref name="bar"/> மொழி'
        assert _clean_wiki_text(text) == "தமிழ் மொழி"

    def test_removes_comments(self) -> None:
        text = "தமிழ்<!-- hidden --> மொழி"
        assert _clean_wiki_text(text) == "தமிழ் மொழி"

    def test_removes_bold_italic_markers(self) -> None:
        text = "'''தமிழ்''' ''மொழி''"
        assert _clean_wiki_text(text) == "தமிழ் மொழி"

    def test_removes_references_section(self) -> None:
        text = "Main content.\n\n== References ==\nSome refs\n* ref1"
        result = _clean_wiki_text(text)
        assert "Main content." in result
        assert "ref1" not in result

    def test_removes_categories(self) -> None:
        text = "Content.\n[[Category:Tamil language]]"
        result = _clean_wiki_text(text)
        assert "Category" not in result

    def test_collapses_newlines(self) -> None:
        text = "Line one.\n\n\n\n\nLine two."
        result = _clean_wiki_text(text)
        assert "\n\n\n" not in result

    def test_empty_input(self) -> None:
        assert _clean_wiki_text("") == ""


class TestWikipediaCollector:
    def test_name(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        assert collector.name == "wikipedia"

    def test_config_defaults(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        assert collector.language == "ta"
        assert collector.batch_size == 10

    @patch("requests.Session")
    @patch("wikipediaapi.Wikipedia")
    def test_collect_yields_documents(
        self, mock_wiki_cls: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
    ) -> None:
        # Mock the allpages API response
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "allpages": [
                    {"title": "தமிழ்", "pageid": 1},
                    {"title": "இலங்கை", "pageid": 2},
                ]
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        # Mock the wikipedia-api page objects
        mock_wiki = MagicMock()
        mock_wiki_cls.return_value = mock_wiki

        mock_page1 = MagicMock()
        mock_page1.exists.return_value = True
        mock_page1.text = "தமிழ் ஒரு திராவிட மொழி. " * 5

        mock_page2 = MagicMock()
        mock_page2.exists.return_value = True
        mock_page2.text = "இலங்கை ஒரு தீவு நாடு. " * 5

        mock_wiki.page.side_effect = [mock_page1, mock_page2]

        collector = _make_collector(tmp_path)
        docs = list(collector.collect(limit=10))

        assert len(docs) == 2
        assert docs[0].source == "wikipedia"
        assert docs[0].license == "CC-BY-SA-4.0"
        assert "தமிழ்" in docs[0].meta["title"]

    def test_checkpoint_integration(self, tmp_path: Path) -> None:
        collector = _make_collector(tmp_path)
        assert collector._read_checkpoint("last_title") is None

        collector._write_checkpoint("last_title", "தமிழ்")
        assert collector._read_checkpoint("last_title") == "தமிழ்"

    @patch("requests.Session")
    @patch("wikipediaapi.Wikipedia")
    def test_skips_short_articles(
        self, mock_wiki_cls: MagicMock, mock_session_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {"allpages": [{"title": "Short", "pageid": 1}]},
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response

        mock_wiki = MagicMock()
        mock_wiki_cls.return_value = mock_wiki
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.text = "Too short"
        mock_wiki.page.return_value = mock_page

        collector = _make_collector(tmp_path)
        docs = list(collector.collect())

        assert len(docs) == 0
