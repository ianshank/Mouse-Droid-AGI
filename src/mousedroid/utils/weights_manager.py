"""Weight management utilities — download, cache, and validate BDI model weights.

Supports downloading trained model weights from HuggingFace Hub with retry
logic and graceful degradation if hf_hub not installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from pathlib import Path

from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

_PROTECTED_DOWNLOAD_ROOTS: tuple[Path, ...] = (
    Path("/etc"),
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/bin"),
    Path("/sbin"),
    Path("/boot"),
    Path("/root"),
    Path("C:/Windows"),
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
)

# ---------------------------------------------------------------------------
# HuggingFace Integration
# ---------------------------------------------------------------------------

_HF_HUB_AVAILABLE = False


def _validate_download_directory(path: Path) -> None:
    """Reject directories that resolve inside protected system locations."""
    for protected_root in _PROTECTED_DOWNLOAD_ROOTS:
        if path.is_relative_to(protected_root):
            raise ValueError(
                f"refusing to write HuggingFace downloads under protected path '{path}'"
            )


_hf_hub_download_impl: Callable[..., str] | None = None


def _missing_hf_hub_download(**kwargs: object) -> str:
    """Stub used when ``huggingface_hub`` is unavailable."""
    raise ImportError(
        "huggingface_hub is not installed. "
        "Install with: pip install 'mousedroid[llm]' or pip install huggingface-hub"
    )


try:
    from huggingface_hub import hf_hub_download as _imported_hf_hub_download

    _hf_hub_download_impl = _imported_hf_hub_download
    _HF_HUB_AVAILABLE = True
except ImportError:
    pass


def _hf_hub_download(**kwargs: object) -> str:
    """Typed wrapper that dispatches to HF Hub when available."""
    if _hf_hub_download_impl is None:
        return _missing_hf_hub_download(**kwargs)
    return _hf_hub_download_impl(**kwargs)


def download_weights_from_huggingface(
    repo_id: str,
    filenames: list[str],
    cache_dir: Path | str,
    *,
    subfolder: str = "",
    local_dir: Path | str | None = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> bool:
    """Download model weights from HuggingFace Hub with retry logic.

    Attempts to download a list of files (e.g., ['belief.npz', 'desire.npz'])
    from a HuggingFace repository. Uses exponential backoff for retries.
    Gracefully degrades if hf_hub not installed.

    When *subfolder* and *local_dir* are provided the files are fetched from
    ``<subfolder>/<filename>`` inside the repo and written directly into
    ``<local_dir>/<subfolder>/`` so that callers find them at
    ``<local_dir>/<subfolder>/<filename>`` (i.e. if local_dir is the parent of
    the intended weights_dir the files land exactly where NeuralBDI expects).

    Args:
        repo_id: HuggingFace repository ID (e.g., "ianshank/mousedroid-weights").
        filenames: List of filenames to download from the repo.
        cache_dir: Local directory to cache downloaded files (used when local_dir is None).
        subfolder: Subfolder within the repo where files live (e.g. ``"bdi"``).
        local_dir: If given, files are written here in flat repo structure instead
            of the HF cache layout.
        max_retries: Maximum retry attempts per file.
        backoff_base: Exponential backoff base (wait = backoff_base ^ attempt).

    Returns:
        True if all files downloaded successfully, False otherwise.
    """
    if not _HF_HUB_AVAILABLE:
        _log.warning(
            "huggingface_hub_not_installed",
            repo_id=repo_id,
            filenames=filenames,
            hint="Install via: pip install 'mousedroid[llm]' or pip install huggingface-hub",
        )
        return False

    cache_dir = Path(cache_dir).resolve()
    _validate_download_directory(cache_dir)
    if local_dir is not None:
        local_dir_path = Path(local_dir).resolve()
        expected_target_dir = (
            (local_dir_path / subfolder).resolve() if subfolder else local_dir_path
        )
        if expected_target_dir != cache_dir:
            raise ValueError(
                "local_dir and subfolder must resolve to cache_dir; "
                f"got {expected_target_dir} != {cache_dir}"
            )
        _validate_download_directory(local_dir_path)
        local_dir = local_dir_path  # use resolved path from here on

    cache_dir.mkdir(parents=True, exist_ok=True)
    if local_dir is not None:
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    all_success = True
    for filename in filenames:
        success = _download_file_with_retry(
            repo_id,
            filename,
            cache_dir,
            subfolder=subfolder,
            local_dir=local_dir,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        if not success:
            all_success = False
            _log.warning(
                "failed_to_download_weight_file",
                repo_id=repo_id,
                filename=filename,
                max_retries=max_retries,
            )

    if all_success:
        _log.info(
            "weights_downloaded_from_huggingface",
            repo_id=repo_id,
            file_count=len(filenames),
            cache_dir=str(cache_dir),
        )
    return all_success


def _download_file_with_retry(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    *,
    subfolder: str = "",
    local_dir: Path | str | None = None,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> bool:
    """Download a single file from HuggingFace with exponential backoff retry.

    Args:
        repo_id: HuggingFace repository ID.
        filename: Single filename to download.
        cache_dir: Local directory to cache the file (used when local_dir is None).
        subfolder: Subfolder within the repo where the file lives.
        local_dir: If given, replicate the repo file structure here instead of
            the HF cache layout.
        max_retries: Maximum retry attempts.
        backoff_base: Exponential backoff base.

    Returns:
        True if download succeeded, False otherwise.
    """
    hf_kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": "model",
    }
    if subfolder:
        hf_kwargs["subfolder"] = subfolder
    if local_dir is not None:
        hf_kwargs["local_dir"] = str(local_dir)
    else:
        hf_kwargs["cache_dir"] = str(cache_dir)

    for attempt in range(max_retries):
        try:
            local_path = _hf_hub_download(**hf_kwargs)
            _log.debug(
                "downloaded_weight_file",
                repo_id=repo_id,
                filename=filename,
                local_path=local_path,
                attempt=attempt + 1,
            )
            return True
        except Exception as e:  # pylint: disable=broad-except
            if attempt < max_retries - 1:
                wait_time = backoff_base**attempt
                _log.debug(
                    "download_retry",
                    repo_id=repo_id,
                    filename=filename,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_seconds=wait_time,
                    error=str(e),
                )
                time.sleep(wait_time)
            else:
                _log.debug(
                    "download_failed_final_attempt",
                    repo_id=repo_id,
                    filename=filename,
                    attempt=attempt + 1,
                    error=str(e),
                )
    return False


# ---------------------------------------------------------------------------
# Tier C1 — SHA-256 integrity verification helper
# ---------------------------------------------------------------------------

# Chunk size for streaming SHA-256 over a file. Picked to balance loop
# overhead against memory pressure on the Jetson Orin Nano. Not a tunable
# the operator should touch; hash output is independent of the chunk size.
_SHA256_CHUNK_BYTES: int = 64 * 1024


def verify_sha256(
    local_path: Path | str,
    expected_hex: str,
    *,
    log_event_prefix: str = "weights",
) -> bool:
    """Verify that ``local_path`` matches the SHA-256 digest ``expected_hex``.

    Safety-critical helper: returns ``True`` only when the file exists, the
    expected digest is a syntactically valid 64-char lowercase hex string,
    and the computed digest matches. Any failure path returns ``False`` and
    emits a structured log event with prefix ``log_event_prefix`` so the
    caller can correlate the failure with the upstream subsystem (e.g.
    ``"cloud_weight_update"`` for the OTA poller).

    The expected digest is supplied by the caller (typically read from a
    ``sha256.txt`` manifest in the same HuggingFace repo as the artifact).
    NEVER hardcoded inside this helper — Tier C1 requirement.

    Args:
        local_path: Path to the local file to verify.
        expected_hex: Hex-encoded SHA-256 digest the file MUST match
            (case-insensitive; whitespace is stripped). Must be exactly
            64 hex characters after normalisation.
        log_event_prefix: Structured-log event-name prefix. The helper emits
            ``<prefix>_sha256_verified`` on success and
            ``<prefix>_sha256_mismatch`` / ``<prefix>_sha256_invalid_input``
            on failure.

    Returns:
        ``True`` iff the file exists, ``expected_hex`` is a valid 64-char
        hex string, and the computed SHA-256 digest matches.
    """
    path = Path(local_path)
    if not path.is_file():
        _log.warning(
            f"{log_event_prefix}_sha256_invalid_input",
            reason="missing_file",
            local_path=str(path),
        )
        return False

    expected = expected_hex.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        _log.warning(
            f"{log_event_prefix}_sha256_invalid_input",
            reason="malformed_expected_digest",
            local_path=str(path),
            expected_len=len(expected),
        )
        return False

    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_SHA256_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    computed = hasher.hexdigest()

    if computed != expected:
        _log.warning(
            f"{log_event_prefix}_sha256_mismatch",
            local_path=str(path),
            expected=expected,
            computed=computed,
        )
        return False

    _log.info(
        f"{log_event_prefix}_sha256_verified",
        local_path=str(path),
        sha256=computed,
    )
    return True


def weights_exist_locally(weights_dir: Path | str, filenames: list[str]) -> bool:
    """Check if all weight files exist locally.

    Args:
        weights_dir: Directory to check.
        filenames: List of filenames to verify.

    Returns:
        True if all files exist, False otherwise.
    """
    weights_dir = Path(weights_dir)
    return all((weights_dir / filename).exists() for filename in filenames)


async def download_weights_async(
    repo_id: str,
    filenames: list[str],
    cache_dir: Path | str,
    *,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> bool:
    """Async wrapper for download_weights_from_huggingface.

    Runs the blocking download in a worker thread to avoid blocking the
    asyncio event loop during startup.

    Args:
        repo_id: HuggingFace repository ID.
        filenames: List of filenames to download.
        cache_dir: Local directory to cache downloaded files.
        max_retries: Maximum retry attempts per file.
        backoff_base: Exponential backoff base.

    Returns:
        True if all files downloaded successfully, False otherwise.
    """
    return await asyncio.to_thread(
        download_weights_from_huggingface,
        repo_id,
        filenames,
        cache_dir,
        max_retries=max_retries,
        backoff_base=backoff_base,
    )
