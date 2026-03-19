"""Weight management utilities — download, cache, and validate BDI model weights.

Supports downloading trained model weights from HuggingFace Hub with retry
logic and graceful degradation if hf_hub not installed.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace Integration
# ---------------------------------------------------------------------------

_HF_HUB_AVAILABLE = False
_hf_hub_download: Callable[..., str]
try:
    from huggingface_hub import hf_hub_download as _hf_hub_download

    _HF_HUB_AVAILABLE = True
except ImportError:

    def _hf_hub_download(**kwargs: object) -> str:
        """Stub — never called when _HF_HUB_AVAILABLE is False."""
        raise ImportError("huggingface_hub is not installed")


def download_weights_from_huggingface(
    repo_id: str,
    filenames: list[str],
    cache_dir: Path | str,
    *,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> bool:
    """Download model weights from HuggingFace Hub with retry logic.

    Attempts to download a list of files (e.g., ['belief.npz', 'desire.npz'])
    from a HuggingFace repository. Uses exponential backoff for retries.
    Gracefully degrades if hf_hub not installed.

    Args:
        repo_id: HuggingFace repository ID (e.g., "ianshank/mousedroid-weights").
        filenames: List of filenames to download from the repo.
        cache_dir: Local directory to cache downloaded files.
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
            hint="Install via: pip install huggingface-hub",
        )
        return False

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_success = True
    for filename in filenames:
        success = _download_file_with_retry(
            repo_id,
            filename,
            cache_dir,
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
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> bool:
    """Download a single file from HuggingFace with exponential backoff retry.

    Args:
        repo_id: HuggingFace repository ID.
        filename: Single filename to download.
        cache_dir: Local directory to cache the file.
        max_retries: Maximum retry attempts.
        backoff_base: Exponential backoff base.

    Returns:
        True if download succeeded, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            local_path = _hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir),
                repo_type="model",
            )
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
