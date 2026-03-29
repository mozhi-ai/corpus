"""Format processed JSONL data into Parquet and generate dataset cards."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mozhi.models import CorpusStats

logger = logging.getLogger(__name__)


def jsonl_to_parquet(input_path: Path, output_path: Path) -> int:
    """Convert a JSONL file to Parquet format.

    Returns the number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict[str, object]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                # Flatten: drop meta dict for Parquet, keep key fields
                rows.append({
                    "text": data["text"],
                    "source": data["source"],
                    "url": data.get("url", ""),
                    "license": data.get("license", ""),
                    "date_collected": data.get("date_collected", ""),
                    "language_score": float(data.get("language_score", 0.0)),
                    "quality_score": float(data.get("quality_score", 0.0)),
                    "domain": data.get("meta", {}).get("domain", ""),
                    "register": data.get("meta", {}).get("register", ""),
                })

    if not rows:
        logger.warning("No rows to convert")
        return 0

    table = pa.Table.from_pylist(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Wrote %d rows to %s", len(rows), output_path)
    return len(rows)


def generate_dataset_card(stats: CorpusStats, repo_id: str) -> str:
    """Generate a HuggingFace dataset card (README.md) with stats leaderboard."""
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    source_rows = ""
    for source, count in sorted(stats.sources.items(), key=lambda x: -x[1]):
        source_rows += f"| {source} | {count:,} |\n"

    license_rows = ""
    for lic, count in sorted(stats.licenses.items(), key=lambda x: -x[1]):
        license_rows += f"| {lic} | {count:,} |\n"

    return f"""---
language:
  - ta
license: cc-by-sa-4.0
task_categories:
  - text-generation
  - fill-mask
tags:
  - tamil
  - corpus
  - nlp
  - llm-training
  - indic
  - dravidian
pretty_name: "mozhi-ai Tamil Corpus"
size_categories:
  - {_size_category(stats.total_documents)}
---

# mozhi-ai Tamil Corpus

A continuously-updated, high-quality Tamil language corpus for LLM training.
Collects text from classical literature, Wikipedia, news, and web sources.

## Corpus Stats (updated {today})

| Metric | Value |
|--------|-------|
| **Total Documents** | {stats.total_documents:,} |
| **Total Characters** | {stats.total_characters:,} |
| **Est. Words** | {stats.total_words:,} |
| **Deduplicated** | {stats.dedup_count:,} removed |
| **Quality Filtered** | {stats.filtered_count:,} removed |
| **Last Updated** | {today} |

### Sources Breakdown

| Source | Documents |
|--------|-----------|
{source_rows}
### Licenses

| License | Documents |
|---------|-----------|
{license_rows}
## Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The Tamil text content |
| `source` | string | Data source identifier |
| `url` | string | Source URL (when available) |
| `license` | string | License of the source data |
| `date_collected` | string | Collection date (ISO 8601) |
| `language_score` | float | fastText Tamil confidence (0-1) |
| `quality_score` | float | Quality heuristic score (0-1) |
| `domain` | string | Content domain (news, literature, etc.) |
| `register` | string | Language register (classical, formal, colloquial) |

## Processing Pipeline

Every document passes through:

1. **Language Detection** — fastText `lid.176.bin` (threshold: 0.7)
2. **Unicode Normalization** — NFKC + Tamil-specific normalization (indic_nlp_library)
3. **Exact Deduplication** — SHA-256 content hashing
4. **Quality Filtering** — length, Tamil character ratio, uniqueness, boilerplate detection

## Sources

- **IndicCorpV2** (AI4Bharat) — Tamil news and articles (CC-0)
- **allenai/c4** — Web-crawled Tamil text (ODC-BY)
- **Tamil Wikipedia** — Encyclopedia articles (CC-BY-SA-4.0)
- **Project Madurai** — Classical Tamil literature (Public Domain)

## Usage

```python
from datasets import load_dataset

# Load the full dataset
ds = load_dataset("{repo_id}")

# Stream without downloading
ds = load_dataset("{repo_id}", streaming=True)

# Access the text
for example in ds["train"]:
    print(example["text"][:100])
```

## Citation

If you use this corpus, please cite:

```bibtex
@misc{{mozhi-ai-tamil-corpus,
  title={{mozhi-ai Tamil Corpus}},
  author={{mozhi-ai contributors}},
  year={{2026}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```

## Contributing

This is an open project. Contributions welcome at [GitHub](https://github.com/mozhi-ai).

---

*Built with [mozhi-ai](https://github.com/mozhi-ai) — automated Tamil corpus pipeline*
"""


def _size_category(n: int) -> str:
    """Map document count to HuggingFace size category."""
    if n < 1_000:
        return "n<1K"
    if n < 10_000:
        return "1K<n<10K"
    if n < 100_000:
        return "10K<n<100K"
    if n < 1_000_000:
        return "100K<n<1M"
    if n < 10_000_000:
        return "1M<n<10M"
    return "n>10M"
