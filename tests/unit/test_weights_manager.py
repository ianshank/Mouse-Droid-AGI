"""Tests for weights_manager — HuggingFace download, retry logic, caching."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mousedroid.utils.weights_manager import (
    _HF_HUB_AVAILABLE,
    _hf_hub_download,
    download_weights_from_huggingface,
    weights_exist_locally,
)


def test_weights_exist_locally_all_present(tmp_path):
    """Test weights_exist_locally returns True when all files present."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()

    # Create weight files
    (weights_dir / "belief.npz").touch()
    (weights_dir / "desire.npz").touch()

    assert weights_exist_locally(weights_dir, ["belief.npz", "desire.npz"]) is True


def test_weights_exist_locally_some_missing(tmp_path):
    """Test weights_exist_locally returns False when some files missing."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()

    # Create only one file
    (weights_dir / "belief.npz").touch()

    assert weights_exist_locally(weights_dir, ["belief.npz", "desire.npz"]) is False


def test_weights_exist_locally_none_present(tmp_path):
    """Test weights_exist_locally returns False when no files present."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()

    assert weights_exist_locally(weights_dir, ["belief.npz", "desire.npz"]) is False


def test_weights_exist_locally_nonexistent_dir():
    """Test weights_exist_locally returns False for nonexistent directory."""
    assert weights_exist_locally("/nonexistent/path", ["belief.npz"]) is False


def test_weights_exist_locally_string_path(tmp_path):
    """Test weights_exist_locally accepts string paths."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "belief.npz").touch()

    # Pass as string instead of Path
    assert weights_exist_locally(str(weights_dir), ["belief.npz"]) is True


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", False)
def test_download_weights_hf_not_available():
    """Test download gracefully fails when huggingface_hub not installed."""
    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz"],
        cache_dir="/tmp/cache",
    )
    assert result is False


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_success(mock_download, tmp_path):
    """Test successful weight download."""
    mock_download.return_value = str(tmp_path / "belief.npz")

    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz"],
        cache_dir=tmp_path,
    )
    assert result is True
    mock_download.assert_called_once()


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager.time.sleep", return_value=None)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_partial_failure(mock_download, mock_sleep, tmp_path):
    """Test partial failure when some files fail to download."""
    # First file succeeds, second always fails (across all retries)
    mock_download.side_effect = [
        str(tmp_path / "belief.npz"),
        RuntimeError("network error"),
        RuntimeError("network error"),
        RuntimeError("network error"),
    ]

    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz", "desire.npz"],
        cache_dir=tmp_path,
        max_retries=3,
    )
    assert result is False
    # 1 call for belief.npz + 3 retries for desire.npz
    assert mock_download.call_count == 4


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager.time.sleep", return_value=None)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_retry_logic(mock_download, mock_sleep, tmp_path):
    """Test retry logic with exponential backoff."""
    # Fail twice, succeed on third attempt
    mock_download.side_effect = [
        RuntimeError("network error"),
        RuntimeError("network error"),
        str(tmp_path / "belief.npz"),
    ]

    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz"],
        cache_dir=tmp_path,
        max_retries=3,
        backoff_base=2.0,
    )
    assert result is True
    assert mock_download.call_count == 3


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager.time.sleep", return_value=None)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_max_retries_exceeded(mock_download, mock_sleep, tmp_path):
    """Test download fails after max retries exceeded."""
    mock_download.side_effect = RuntimeError("persistent network error")

    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz"],
        cache_dir=tmp_path,
        max_retries=2,
    )
    assert result is False
    assert mock_download.call_count == 2


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_multiple_files(mock_download, tmp_path):
    """Test downloading multiple weight files."""
    filenames = ["belief.npz", "desire.npz", "intention.npz"]
    mock_download.side_effect = [str(tmp_path / f) for f in filenames]

    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=filenames,
        cache_dir=tmp_path,
    )
    assert result is True
    assert mock_download.call_count == len(filenames)


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_creates_cache_dir(mock_download, tmp_path):
    """Test download creates cache directory if it doesn't exist."""
    cache_dir = tmp_path / "nonexistent" / "cache"
    mock_download.return_value = str(cache_dir / "belief.npz")

    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz"],
        cache_dir=cache_dir,
    )
    assert result is True
    assert cache_dir.exists()


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_repo_id_passed(mock_download, tmp_path):
    """Test correct repo_id is passed to hf_hub_download."""
    mock_download.return_value = str(tmp_path / "belief.npz")

    download_weights_from_huggingface(
        repo_id="custom/repo",
        filenames=["belief.npz"],
        cache_dir=tmp_path,
    )

    # Verify repo_id was passed correctly
    call_kwargs = mock_download.call_args[1]
    assert call_kwargs["repo_id"] == "custom/repo"


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_cache_dir_string(mock_download, tmp_path):
    """Test cache_dir can be passed as string."""
    mock_download.return_value = str(tmp_path / "belief.npz")

    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz"],
        cache_dir=str(tmp_path),  # String instead of Path
    )
    assert result is True


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
async def test_download_weights_async(mock_download, tmp_path):
    """Test async wrapper delegates to sync download in thread."""
    from mousedroid.utils.weights_manager import download_weights_async

    mock_download.return_value = str(tmp_path / "belief.npz")

    result = await download_weights_async(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz"],
        cache_dir=tmp_path,
    )
    assert result is True
    mock_download.assert_called_once()


def test_hf_hub_download_stub_raises_when_not_available():
    """Cover line 34: the fallback stub raises ImportError when called."""
    if _HF_HUB_AVAILABLE:
        pytest.skip("huggingface_hub is installed; stub not defined")
    with pytest.raises(ImportError, match="huggingface_hub is not installed"):
        _hf_hub_download(repo_id="test", filename="test.npz")
