"""Tests for mozhi.config."""

from pathlib import Path

import pytest

from mozhi.config import CollectorConfig, MozhiConfig, PipelineConfig, load_config


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    config_content = """\
corpus_dir: data

collectors:
  - name: huggingface
    enabled: true
    params:
      datasets:
        - repo: statmt/cc100
          subset: ta
  - name: wikipedia
    enabled: false
    params:
      language: ta

pipeline:
  min_language_score: 0.8
  min_text_length: 100

publish:
  repo_id: test-user/test-corpus
  private: true
"""
    path = tmp_path / "config.yaml"
    path.write_text(config_content)
    return path


def test_load_config(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.corpus_dir == Path("data")
    assert len(config.collectors) == 2
    assert config.collectors[0].name == "huggingface"
    assert config.collectors[1].name == "wikipedia"
    assert config.collectors[1].enabled is False


def test_pipeline_config_override(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.pipeline.min_language_score == 0.8
    assert config.pipeline.min_text_length == 100
    # Defaults preserved for unset values
    assert config.pipeline.max_text_length == 100_000


def test_publish_config(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.publish.repo_id == "test-user/test-corpus"
    assert config.publish.private is True


def test_enabled_collectors(config_file: Path) -> None:
    config = load_config(config_file)
    enabled = config.enabled_collectors()
    assert len(enabled) == 1
    assert enabled[0].name == "huggingface"


def test_get_collector(config_file: Path) -> None:
    config = load_config(config_file)
    hf = config.get_collector("huggingface")
    assert hf is not None
    assert hf.name == "huggingface"
    assert config.get_collector("nonexistent") is None


def test_load_empty_config(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    config = load_config(path)
    assert config.collectors == []
    assert config.pipeline == PipelineConfig()


def test_collector_config_defaults() -> None:
    c = CollectorConfig(name="test")
    assert c.enabled is True
    assert c.params == {}


def test_mozhi_config_defaults() -> None:
    config = MozhiConfig()
    assert config.corpus_dir == Path("corpus")
    assert config.collectors == []
