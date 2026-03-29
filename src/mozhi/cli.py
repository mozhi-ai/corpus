"""CLI entry point for the mozhi corpus pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from mozhi.config import load_config
from mozhi.utils.io import ensure_dirs
from mozhi.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from mozhi.models import Document

logger = get_logger("cli")

DEFAULT_CONFIG = Path("config.yaml")


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_CONFIG,
    help="Path to config.yaml",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
@click.pass_context
def main(ctx: click.Context, config_path: Path, log_level: str) -> None:
    """mozhi-ai: Tamil language corpus collection pipeline."""
    setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)
    ensure_dirs(ctx.obj["config"].corpus_dir)


@main.command()
@click.option("--source", help="Collect from a specific source only")
@click.option("--limit", type=int, help="Limit number of documents to collect")
@click.pass_context
def collect(ctx: click.Context, source: str | None, limit: int | None) -> None:
    """Collect raw text from configured sources."""
    config = ctx.obj["config"]
    collectors_to_run = config.enabled_collectors()

    if source:
        collectors_to_run = [c for c in collectors_to_run if c.name == source]
        if not collectors_to_run:
            raise click.ClickException(f"Unknown or disabled source: {source}")

    logger.info(
        "Starting collection from %d source(s): %s",
        len(collectors_to_run),
        ", ".join(c.name for c in collectors_to_run),
    )

    from datetime import UTC, datetime

    from mozhi.collectors import get_collector
    from mozhi.utils.io import write_jsonl

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    total = 0

    for collector_config in collectors_to_run:
        logger.info("Collecting from %s...", collector_config.name)
        collector = get_collector(collector_config, config.corpus_dir)
        docs = collector.collect(limit=limit)
        output_path = config.corpus_dir / "raw" / collector.name / f"{today}.jsonl"
        count = write_jsonl(docs, output_path)
        total += count
        click.echo(f"  {collector.name}: collected {count:,} documents → {output_path}")

    click.echo(f"Total: {total:,} documents collected")


@main.command()
@click.pass_context
def process(ctx: click.Context) -> None:
    """Run the processing pipeline on raw data."""
    config = ctx.obj["config"]
    logger.info("Processing pipeline starting")

    from datetime import UTC, datetime

    from mozhi.models import CorpusStats
    from mozhi.processing.pipeline import Pipeline
    from mozhi.utils.io import read_jsonl, write_jsonl

    pipeline = Pipeline.default(config.pipeline)

    # Gather all raw JSONL files
    raw_dir = config.corpus_dir / "raw"
    raw_files = sorted(raw_dir.glob("**/*.jsonl"))
    if not raw_files:
        click.echo("No raw data found. Run 'mozhi collect' first.")
        return

    click.echo(f"Found {len(raw_files)} raw file(s) to process")

    # Stream all raw docs through the pipeline
    def _all_raw_docs() -> Iterator[Document]:
        for f in raw_files:
            yield from read_jsonl(f)

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    output_path = config.corpus_dir / "processed" / f"{today}.jsonl"
    stats = CorpusStats()

    processed_docs = pipeline.run(_all_raw_docs())

    def _track_stats(docs: Iterable[Document]) -> Iterator[Document]:
        for doc in docs:
            stats.update(doc)
            yield doc

    count = write_jsonl(_track_stats(processed_docs), output_path)
    click.echo(f"Processed {count:,} documents → {output_path}")
    click.echo(stats.summary())


@main.command()
@click.option("--repo", help="Override HuggingFace repo ID")
@click.option("--private/--public", default=None, help="Override repo visibility")
@click.pass_context
def publish(ctx: click.Context, repo: str | None, private: bool | None) -> None:
    """Publish processed corpus to HuggingFace Hub."""
    import os

    from mozhi.models import CorpusStats
    from mozhi.publishing.formatter import generate_dataset_card, jsonl_to_parquet
    from mozhi.publishing.uploader import push_to_hub

    config = ctx.obj["config"]
    repo_id = repo or config.publish.repo_id
    is_private = private if private is not None else config.publish.private

    # Check for HF token
    if not os.environ.get("HF_TOKEN"):
        raise click.ClickException(
            "HF_TOKEN environment variable not set. "
            "Get a token at https://huggingface.co/settings/tokens"
        )

    # Find the latest processed JSONL
    processed_dir = config.corpus_dir / "processed"
    processed_files = sorted(processed_dir.glob("*.jsonl"), reverse=True)
    if not processed_files:
        raise click.ClickException("No processed data found. Run 'mozhi process' first.")

    latest = processed_files[0]
    click.echo(f"Publishing {latest.name} to {repo_id}")

    # Convert to Parquet
    parquet_path = config.corpus_dir / "published" / latest.name.replace(".jsonl", ".parquet")
    row_count = jsonl_to_parquet(latest, parquet_path)
    click.echo(f"  Converted to Parquet: {row_count:,} rows")

    # Compute stats for dataset card
    from mozhi.utils.io import read_jsonl

    stats = CorpusStats()
    for doc in read_jsonl(latest):
        stats.update(doc)

    # Generate dataset card
    card = generate_dataset_card(stats, repo_id)

    # Push to Hub
    from datetime import UTC, datetime

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    commit_msg = f"{config.publish.commit_message_prefix} ({today}, {row_count:,} docs)"

    url = push_to_hub(
        parquet_path,
        repo_id,
        private=is_private,
        commit_message=commit_msg,
        dataset_card=card,
    )
    click.echo(f"  Published to {url}")
    click.echo(f"  {stats.summary()}")


@main.command(name="run-all")
@click.option("--limit", type=int, help="Limit documents per source")
@click.pass_context
def run_all(ctx: click.Context, limit: int | None) -> None:
    """Run the full pipeline: collect → process → publish."""
    logger.info("Starting full pipeline run")
    ctx.invoke(collect, source=None, limit=limit)
    ctx.invoke(process)
    ctx.invoke(publish, repo=None, private=None)
    logger.info("Full pipeline run complete")
