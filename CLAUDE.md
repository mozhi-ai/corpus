# mozhi-ai

Tamil language corpus collection pipeline for LLM training. Collects, cleans, deduplicates, and publishes Tamil text to HuggingFace Hub daily.

## Tech Stack

- Python 3.11+, uv for package management
- Click CLI, PyYAML config
- HuggingFace datasets/hub for data access and publishing
- fastText for language detection, indic_nlp_library for Tamil normalization
- datasketch for MinHash LSH deduplication

## Project Structure

- `src/mozhi/` — main package (collectors, processing, publishing, utils)
- `tests/` — pytest test suite mirroring src/ structure
- `corpus/` — raw/, processed/, published/, dedup_index/
- `config.yaml` — pipeline configuration
- `scripts/` — utility scripts
- `notebooks/` — exploration notebooks

## Commands

- `uv run mozhi --help` — CLI entry point
- `uv run mozhi collect --source huggingface` — collect from a source
- `uv run mozhi process` — run processing pipeline
- `uv run mozhi publish` — push to HuggingFace Hub
- `uv run mozhi run-all` — full pipeline
- `uv run pytest` — run tests
- `uv run pytest tests/test_foo.py -k test_name` — run a single test
- `uv run ruff check .` — lint
- `uv run ruff format .` — format

## Code Style

- Use type hints everywhere
- Prefer dataclasses over raw dicts
- Use pathlib for file paths, not os.path
- Google Python style docstrings only for public APIs
- Documents flow as `Iterator[Document]` — always stream, never materialize full corpus in memory

## Workflow

- Always run tests after code changes
- Use conventional commits: feat:, fix:, docs:, refactor:, test:
- Write tests before marking a task complete
- Prefer running single tests, not the whole suite, for speed

## Skills

- See @.claude/skills/ for domain-specific workflows
