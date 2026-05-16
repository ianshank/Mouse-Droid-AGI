"""Tests for weights_manager — HuggingFace download, retry logic, caching."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from mousedroid.utils.weights_manager import (
    _HF_HUB_AVAILABLE,
    _hf_hub_download,
    download_weights_from_huggingface,
    verify_sha256,
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
def test_download_weights_subfolder_local_dir_must_resolve_to_cache_dir(mock_download, tmp_path):
    """Reject local_dir/subfolder combinations that write outside cache_dir."""
    with pytest.raises(ValueError, match="must resolve to cache_dir"):
        download_weights_from_huggingface(
            repo_id="ianshank/mousedroid-weights",
            filenames=["belief.npz"],
            cache_dir=tmp_path / "weights" / "bdi",
            subfolder="bdi",
            local_dir=tmp_path,
        )
    mock_download.assert_not_called()


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
@patch("mousedroid.utils.weights_manager._hf_hub_download", create=True)
def test_download_weights_subfolder_local_dir_valid(mock_download, tmp_path):
    """Allow the local_dir/subfolder combination used by the factory."""
    weights_dir = tmp_path / "weights" / "bdi"
    local_dir = weights_dir.parent
    mock_download.return_value = str(weights_dir / "belief.npz")

    result = download_weights_from_huggingface(
        repo_id="ianshank/mousedroid-weights",
        filenames=["belief.npz"],
        cache_dir=weights_dir,
        subfolder="bdi",
        local_dir=local_dir,
    )

    assert result is True
    assert mock_download.call_args.kwargs["subfolder"] == "bdi"
    assert mock_download.call_args.kwargs["local_dir"] == str(local_dir.resolve())


@patch("mousedroid.utils.weights_manager._HF_HUB_AVAILABLE", True)
def test_download_weights_rejects_protected_cache_dir(tmp_path, monkeypatch):
    """Reject cache directories that resolve under protected roots before mkdir/download."""
    import mousedroid.utils.weights_manager as weights_manager

    protected_root = (tmp_path / "protected-root").resolve()
    monkeypatch.setattr(weights_manager, "_PROTECTED_DOWNLOAD_ROOTS", (protected_root,))

    with pytest.raises(ValueError, match="protected path"):
        download_weights_from_huggingface(
            repo_id="ianshank/mousedroid-weights",
            filenames=["belief.npz"],
            cache_dir=protected_root / "weights",
        )


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


def test_hf_hub_available_flag_set_when_hub_importable():
    """Cover line 29: _HF_HUB_AVAILABLE = True when huggingface_hub is importable.

    We simulate the import-time branch by reloading the module with a
    fake ``huggingface_hub`` on ``sys.path``.
    """
    import importlib
    import sys
    import types

    # Create a fake huggingface_hub module with a stub hf_hub_download
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.hf_hub_download = lambda **kwargs: "/fake/path"  # type: ignore[attr-defined]

    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = fake_hf
    try:
        # Force reimport of weights_manager to hit the try-branch
        import mousedroid.utils.weights_manager as wm

        importlib.reload(wm)
        assert wm._HF_HUB_AVAILABLE is True
    finally:
        # Restore original state and reload to reset module globals
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved
        importlib.reload(wm)


# ---------------------------------------------------------------------------
# Tier C1 — verify_sha256 helper
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    """Compute the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def test_verify_sha256_match_returns_true(tmp_path):
    """verify_sha256 returns True when the digest matches."""
    payload = b"mousedroid-tier-c1-weight-update"
    f = tmp_path / "weights.bin"
    f.write_bytes(payload)
    assert verify_sha256(f, _sha256(payload)) is True


def test_verify_sha256_mismatch_returns_false(tmp_path):
    """verify_sha256 returns False on a digest mismatch — safety-critical."""
    f = tmp_path / "weights.bin"
    f.write_bytes(b"actual-contents")
    bogus = _sha256(b"different-contents")
    assert verify_sha256(f, bogus) is False


def test_verify_sha256_case_insensitive_and_whitespace_tolerant(tmp_path):
    """Expected digest is normalised (lowercase, stripped)."""
    payload = b"abc"
    f = tmp_path / "weights.bin"
    f.write_bytes(payload)
    digest = _sha256(payload).upper()
    assert verify_sha256(f, f"  {digest}\n") is True


def test_verify_sha256_missing_file_returns_false(tmp_path):
    """verify_sha256 returns False (no crash) when the file is missing."""
    assert verify_sha256(tmp_path / "nope.bin", _sha256(b"anything")) is False


def test_verify_sha256_rejects_malformed_expected(tmp_path):
    """Malformed expected digest is refused without computing the hash."""
    f = tmp_path / "weights.bin"
    f.write_bytes(b"data")
    assert verify_sha256(f, "deadbeef") is False  # too short
    assert verify_sha256(f, "z" * 64) is False  # non-hex
    assert verify_sha256(f, "") is False


def test_verify_sha256_log_event_prefix_propagates(tmp_path, capsys):
    """The log_event_prefix kwarg gates the structured-log event name."""
    f = tmp_path / "weights.bin"
    f.write_bytes(b"data")
    bogus = _sha256(b"other")
    assert verify_sha256(f, bogus, log_event_prefix="cloud_weight_update") is False
    # structlog renders to stdout/stderr; the event-name string appears in
    # the captured output regardless of formatter (key=value / json / dev).
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "cloud_weight_update_sha256_mismatch" in combined


def test_verify_sha256_handles_os_error_on_read(tmp_path, monkeypatch, capsys):
    """OSError during file read must fail closed + log + NOT propagate.

    Regression net for Copilot 3253310002: safety-critical OTA gating must
    NEVER let an OSError from ``path.open('rb').read(...)`` propagate and
    crash the poller. Verified by patching ``Path.open`` to raise OSError
    after the existence check passed.
    """
    from pathlib import Path as _Path

    f = tmp_path / "weights.bin"
    f.write_bytes(b"data")
    digest = _sha256(b"data")

    # Patch Path.open since verify_sha256 calls ``path.open("rb")``.
    # Capture the original via the descriptor protocol so the fake can
    # delegate to it for any unrelated paths (none in this test, but
    # keeping the delegation prevents side-effects if a future
    # implementation does extra reads).
    original_path_open = _Path.open

    def _failing_path_open(self, *args, **kwargs):
        if str(self).endswith("weights.bin") and (args and args[0] == "rb"):
            raise OSError("simulated transient FS error")
        return original_path_open(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "open", _failing_path_open)

    result = verify_sha256(f, digest, log_event_prefix="cloud_weight_update")
    assert result is False, "OSError on read must yield False (fail-closed)"
    captured = capsys.readouterr()
    assert "cloud_weight_update_sha256_invalid_input" in (captured.out + captured.err)
    assert "file_read_failed" in (captured.out + captured.err)
