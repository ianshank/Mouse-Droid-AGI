"""Tests for the shared streaming SHA-256 file digest helper.

digest_file_sha256() previously only had incidental coverage via the growth/
on-device-learning slot-store tests, all of which hash small torch
state-dicts well under the 64 KiB streaming chunk size — so the function's
own multi-iteration streaming-read loop was never actually exercised more
than once, even though line coverage read 100%. This file closes that gap
directly, independent of any slot-store caller.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mousedroid.common.hashing import _SHA256_CHUNK_BYTES, digest_file_sha256


def test_digest_matches_hashlib_reference(tmp_path: Path) -> None:
    path = tmp_path / "small.bin"
    content = b"the quick brown fox jumps over the lazy dog"
    path.write_bytes(content)

    assert digest_file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_digest_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")

    assert digest_file_sha256(path) == hashlib.sha256(b"").hexdigest()


def test_digest_spans_multiple_chunks(tmp_path: Path) -> None:
    """A file larger than _SHA256_CHUNK_BYTES forces the streaming loop to
    actually iterate more than once — the case incidental slot-store
    coverage never reached."""
    # Two full chunks plus a partial third, so the loop runs 3 real reads
    # plus the terminating empty read.
    size = _SHA256_CHUNK_BYTES * 2 + 1000
    content = bytes((i % 256) for i in range(size))
    path = tmp_path / "large.bin"
    path.write_bytes(content)

    assert digest_file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_digest_exact_chunk_boundary(tmp_path: Path) -> None:
    """A file exactly one chunk long — the boundary between a single read
    returning the full chunk and the loop needing to read again."""
    content = bytes((i % 256) for i in range(_SHA256_CHUNK_BYTES))
    path = tmp_path / "exact.bin"
    path.write_bytes(content)

    assert digest_file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_digest_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.bin"

    with pytest.raises(FileNotFoundError):
        digest_file_sha256(missing)
