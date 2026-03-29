"""Upload datasets to HuggingFace Hub."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def push_to_hub(
    parquet_path: Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "corpus: daily update",
    dataset_card: str | None = None,
) -> str:
    """Push a Parquet file to HuggingFace Hub as a dataset.

    Returns the URL of the dataset on the Hub.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Ensure repo exists
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    logger.info("Repo ensured: %s (private=%s)", repo_id, private)

    # Upload the Parquet file with retries
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=f"data/{parquet_path.name}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
            logger.info("Uploaded %s to %s (attempt %d)", parquet_path.name, repo_id, attempt)
            break
        except Exception:
            logger.warning(
                "Upload attempt %d/%d failed for %s",
                attempt,
                MAX_RETRIES,
                parquet_path.name,
                exc_info=True,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY * attempt)

    # Upload dataset card if provided
    if dataset_card:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                api.upload_file(
                    path_or_fileobj=dataset_card.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"{commit_message} — update dataset card",
                )
                logger.info("Dataset card updated for %s", repo_id)
                break
            except Exception:
                logger.warning(
                    "Card upload attempt %d/%d failed",
                    attempt,
                    MAX_RETRIES,
                    exc_info=True,
                )
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_DELAY * attempt)

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Dataset available at %s", url)
    return url
