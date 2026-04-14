"""Smoke tests for the LLM model download script.

Validates that scripts/download_model.sh exists, is executable, and
contains expected safety logic (env var checks, checksum verification,
retry handling).
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "download_model.sh"


# ---------------------------------------------------------------------------
# Script existence and permissions
# ---------------------------------------------------------------------------


def test_download_script_exists() -> None:
    """scripts/download_model.sh must exist."""
    assert _SCRIPT_PATH.exists(), f"Missing: {_SCRIPT_PATH}"


def test_download_script_is_executable() -> None:
    """scripts/download_model.sh must have executable permission."""
    if not _SCRIPT_PATH.exists():
        pytest.skip("download_model.sh not available in worktree")
    mode = _SCRIPT_PATH.stat().st_mode
    assert mode & (
        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    ), "download_model.sh is not executable"


def test_download_script_has_shebang() -> None:
    """Script should start with a proper shebang line."""
    if not _SCRIPT_PATH.exists():
        pytest.skip("download_model.sh not available in worktree")
    first_line = _SCRIPT_PATH.read_text().split("\n", maxsplit=1)[0]
    assert first_line.startswith("#!/"), f"Bad shebang: {first_line}"


# ---------------------------------------------------------------------------
# Content validation — env var configuration
# ---------------------------------------------------------------------------


def _script_text() -> str:
    """Read script contents, skipping if not available."""
    if not _SCRIPT_PATH.exists():
        pytest.skip("download_model.sh not available in worktree")
    return _SCRIPT_PATH.read_text()


def test_script_uses_model_url_env_var() -> None:
    """Script should reference MODEL_URL for configurability."""
    text = _script_text()
    assert "MODEL_URL" in text


def test_script_uses_model_path_env_var() -> None:
    """Script should reference MODEL_PATH for destination configurability."""
    text = _script_text()
    assert "MODEL_PATH" in text


def test_script_uses_model_checksum_env_var() -> None:
    """Script should reference MODEL_CHECKSUM for integrity verification."""
    text = _script_text()
    assert "MODEL_CHECKSUM" in text


def test_script_uses_max_retries_env_var() -> None:
    """Script should reference MAX_RETRIES for download resilience."""
    text = _script_text()
    assert "MAX_RETRIES" in text


# ---------------------------------------------------------------------------
# Checksum verification logic
# ---------------------------------------------------------------------------


def test_script_has_checksum_verification() -> None:
    """Script should contain SHA-256 checksum verification logic."""
    text = _script_text()
    assert "sha256" in text.lower() or "shasum" in text.lower()


def test_script_removes_file_on_checksum_mismatch() -> None:
    """Script should remove downloaded file when checksum fails."""
    text = _script_text()
    assert "rm -f" in text or "rm " in text


# ---------------------------------------------------------------------------
# Download retry logic
# ---------------------------------------------------------------------------


def test_script_has_retry_loop() -> None:
    """Script should implement a retry loop for downloads."""
    text = _script_text()
    assert "retry" in text.lower() or "attempt" in text.lower()


def test_script_supports_wget_and_curl() -> None:
    """Script should support both wget and curl for portability."""
    text = _script_text()
    assert "wget" in text
    assert "curl" in text


# ---------------------------------------------------------------------------
# Safety: set -euo pipefail
# ---------------------------------------------------------------------------


def test_script_uses_strict_mode() -> None:
    """Script should use set -euo pipefail for safety."""
    text = _script_text()
    assert "set -euo pipefail" in text
