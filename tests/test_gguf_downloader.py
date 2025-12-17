import pytest
from pathlib import Path
from unittest.mock import patch

from bookmarks.models.gguf import DEFAULT_FILENAME, DEFAULT_REPO_ID, ensure_gguf_model


def test_ensure_gguf_model_downloads_on_first_call(tmp_path: Path) -> None:
    """test that model is downloaded when not in cache."""
    cache_dir = tmp_path / "models"
    mock_path = tmp_path / "downloaded.gguf"
    mock_path.touch()

    with patch("bookmarks.models.gguf.hf_hub_download") as mock_download:
        mock_download.return_value = str(mock_path)

        result = ensure_gguf_model(
            repo_id="test/repo",
            filename="test.gguf",
            cache_dir=cache_dir,
        )

        assert result == mock_path
        mock_download.assert_called_once_with(
            repo_id="test/repo",
            filename="test.gguf",
            cache_dir=str(cache_dir),
            local_files_only=False,
        )


def test_ensure_gguf_model_uses_defaults(tmp_path: Path) -> None:
    """test that defaults are used when no args provided."""
    mock_path = tmp_path / "default.gguf"
    mock_path.touch()

    with patch("bookmarks.models.gguf.hf_hub_download") as mock_download:
        mock_download.return_value = str(mock_path)

        result = ensure_gguf_model()

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["repo_id"] == DEFAULT_REPO_ID
        assert call_kwargs["filename"] == DEFAULT_FILENAME


def test_ensure_gguf_model_raises_on_download_failure() -> None:
    """test that appropriate error is raised on download failure."""
    from huggingface_hub.utils import HfHubHTTPError

    with patch("bookmarks.models.gguf.hf_hub_download") as mock_download:
        # simulate a download error with proper exception
        mock_download.side_effect = RuntimeError("network error")

        with pytest.raises(RuntimeError, match="unexpected error downloading model"):
            ensure_gguf_model(repo_id="invalid/repo", filename="missing.gguf")


def test_ensure_gguf_model_validates_inputs() -> None:
    """test that empty repo_id or filename raises valueerror."""
    with pytest.raises(ValueError, match="both repo_id and filename are required"):
        ensure_gguf_model(repo_id="", filename="test.gguf")

    with pytest.raises(ValueError, match="both repo_id and filename are required"):
        ensure_gguf_model(repo_id="test/repo", filename="")
