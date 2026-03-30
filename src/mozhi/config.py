"""Configuration loading and typed config dataclasses."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class CollectorConfig:
    """Configuration for a single data collector."""

    name: str
    enabled: bool = True
    params: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""

    min_language_score: float = 0.7
    min_text_length: int = 50
    max_text_length: int = 100_000
    dedup_threshold: float = 0.8
    quality_perplexity_max: float = 5000.0
    tamil_char_ratio_min: float = 0.5


@dataclasses.dataclass
class PublishConfig:
    """Configuration for HuggingFace publishing."""

    repo_id: str = "mozhi-ai/tamil-corpus"
    private: bool = False
    commit_message_prefix: str = "corpus: daily update"


@dataclasses.dataclass
class MozhiConfig:
    """Top-level configuration for the mozhi pipeline."""

    collectors: list[CollectorConfig] = dataclasses.field(default_factory=list)
    pipeline: PipelineConfig = dataclasses.field(default_factory=PipelineConfig)
    publish: PublishConfig = dataclasses.field(default_factory=PublishConfig)
    corpus_dir: Path = Path("corpus")

    def get_collector(self, name: str) -> CollectorConfig | None:
        """Get a collector config by name."""
        for c in self.collectors:
            if c.name == name:
                return c
        return None

    def enabled_collectors(self) -> list[CollectorConfig]:
        """Return only enabled collectors."""
        return [c for c in self.collectors if c.enabled]


def load_config(path: Path) -> MozhiConfig:
    """Load configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    collectors = [
        CollectorConfig(
            name=c["name"],
            enabled=c.get("enabled", True),
            params=c.get("params", {}),
        )
        for c in raw.get("collectors", [])
    ]

    pipeline_raw = raw.get("pipeline", {})
    pipeline = PipelineConfig(
        **{k: v for k, v in pipeline_raw.items() if k in PipelineConfig.__dataclass_fields__}
    )

    publish_raw = raw.get("publish", {})
    publish = PublishConfig(
        **{k: v for k, v in publish_raw.items() if k in PublishConfig.__dataclass_fields__}
    )

    corpus_dir = Path(raw.get("corpus_dir", "corpus"))

    return MozhiConfig(
        collectors=collectors,
        pipeline=pipeline,
        publish=publish,
        corpus_dir=corpus_dir,
    )
