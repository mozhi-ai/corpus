"""Tests for the HuggingFace uploader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mozhi.publishing.uploader import push_to_hub


class TestPushToHub:
    @patch("huggingface_hub.HfApi")
    def test_creates_repo_and_uploads(self, mock_api_cls: MagicMock, tmp_path: Path) -> None:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        parquet = tmp_path / "test.parquet"
        parquet.write_bytes(b"fake parquet data")

        url = push_to_hub(parquet, "user/repo", private=True, commit_message="test upload")

        # Verify repo creation
        mock_api.create_repo.assert_called_once_with(
            repo_id="user/repo",
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )

        # Verify file upload
        mock_api.upload_file.assert_called_once()
        upload_call = mock_api.upload_file.call_args
        assert upload_call.kwargs["repo_id"] == "user/repo"
        assert upload_call.kwargs["path_in_repo"] == "data/test.parquet"

        assert url == "https://huggingface.co/datasets/user/repo"

    @patch("huggingface_hub.HfApi")
    def test_uploads_dataset_card(self, mock_api_cls: MagicMock, tmp_path: Path) -> None:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        parquet = tmp_path / "test.parquet"
        parquet.write_bytes(b"data")

        push_to_hub(parquet, "user/repo", dataset_card="# My Dataset\nContent here")

        # Should have two uploads: parquet + README
        assert mock_api.upload_file.call_count == 2
        second_call = mock_api.upload_file.call_args_list[1]
        assert second_call.kwargs["path_in_repo"] == "README.md"

    @patch("time.sleep")
    @patch("huggingface_hub.HfApi")
    def test_retries_on_failure(
        self, mock_api_cls: MagicMock, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        # Fail first upload, succeed on retry
        mock_api.upload_file.side_effect = [Exception("timeout"), None]

        parquet = tmp_path / "test.parquet"
        parquet.write_bytes(b"data")

        push_to_hub(parquet, "user/repo")

        assert mock_api.upload_file.call_count == 2
        mock_sleep.assert_called_once_with(5)  # RETRY_DELAY * attempt

    @patch("time.sleep")
    @patch("huggingface_hub.HfApi")
    def test_raises_after_max_retries(
        self, mock_api_cls: MagicMock, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.upload_file.side_effect = Exception("persistent failure")

        parquet = tmp_path / "test.parquet"
        parquet.write_bytes(b"data")

        import pytest

        with pytest.raises(Exception, match="persistent failure"):
            push_to_hub(parquet, "user/repo")

        assert mock_api.upload_file.call_count == 3  # MAX_RETRIES
