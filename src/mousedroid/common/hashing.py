"""Shared streaming SHA-256 file digest.

Used by both the growth-pillar and on-device-learning slot stores to
stream-hash a persisted candidate file without loading it fully into memory
(mirrors the C1 OTA weight-update integrity primitive,
:func:`mousedroid.utils.weights_manager.verify_sha256`, ADR-010).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_SHA256_CHUNK_BYTES: int = 64 * 1024  # hardcoded-ok: streaming I/O chunk size, not runtime-tunable


def digest_file_sha256(path: Path) -> str:
    """Stream-hash ``path`` with SHA-256.

    Args:
        path: File to hash.

    Returns:
        The lowercase hex digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_SHA256_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = ["digest_file_sha256"]
