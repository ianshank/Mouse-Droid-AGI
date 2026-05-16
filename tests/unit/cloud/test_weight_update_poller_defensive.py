"""Defensive-path tests for :class:`HuggingFaceWeightUpdatePoller` (Tier C1).

Added in response to PR #94 reviewer feedback (Gemini + Copilot). The
sibling file :file:`test_weight_update_poller.py` covers the happy-path
poll loop; this file targets the new fail-closed branches:

* SHA-256 manifest parsing (empty / unreadable / IndexError).
* Download timeout + retry / backoff (``download_timeout_s``, ``max_retries``).
* Protected-path validation on ``cache_dir``.

Keeping the defensive tests in a dedicated file so the happy-path file
stays narrowly scoped and so future reviewers can see at a glance which
defenses are pinned by tests.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from mousedroid.cloud.weight_update_poller import HuggingFaceWeightUpdatePoller
from mousedroid.config.schema import WeightUpdatePollConfig


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_cfg(
    tmp_path: Path,
    *,
    poll_interval_s: float = 0.0,
    download_timeout_s: float = 30.0,
    max_retries: int = 2,
) -> WeightUpdatePollConfig:
    return WeightUpdatePollConfig(
        poll_interval_s=poll_interval_s,
        download_timeout_s=download_timeout_s,
        max_retries=max_retries,
        policy_repo_id="ianshank/test-policy",
        policy_filename="policy.bin",
        cache_dir=str(tmp_path / "cache"),
        sha256_manifest_filename="sha256.txt",
    )


class _FakeRepoInfo:
    def __init__(self, sha: str) -> None:
        self.sha = sha


def _build_poller_with_overrides(
    tmp_path: Path,
    *,
    hf_api: Any,
    hf_download: Any,
    **cfg_kwargs: Any,
) -> HuggingFaceWeightUpdatePoller:
    cfg = _make_cfg(tmp_path, **cfg_kwargs)
    return HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id=cfg.policy_repo_id,
        filename=cfg.policy_filename,
        engine_type="policy",
        hf_api_factory=lambda: hf_api,
        hf_download=hf_download,
    )


# ---------------------------------------------------------------------------
# Manifest parsing defensive guards (Gemini #2, Copilot 3253293653/3253309989)
# ---------------------------------------------------------------------------


def test_parse_manifest_returns_none_on_empty_file(tmp_path):
    """Empty manifest must fail closed — returns None, never IndexError."""
    manifest_path = tmp_path / "sha256.txt"
    manifest_path.write_text("")
    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=object(),
        hf_download=lambda **_: "",
    )
    assert poller._parse_sha256_manifest(str(manifest_path)) is None


def test_parse_manifest_returns_none_on_whitespace_only_file(tmp_path):
    """Whitespace-only manifest must fail closed (would crash on [0] otherwise)."""
    manifest_path = tmp_path / "sha256.txt"
    manifest_path.write_text("   \n   \t  \n")
    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=object(),
        hf_download=lambda **_: "",
    )
    assert poller._parse_sha256_manifest(str(manifest_path)) is None


def test_parse_manifest_returns_none_on_missing_file(tmp_path):
    """OSError on read must be caught + logged + treated as fail-closed."""
    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=object(),
        hf_download=lambda **_: "",
    )
    assert poller._parse_sha256_manifest(str(tmp_path / "does_not_exist.txt")) is None


def test_parse_manifest_handles_sha256sum_style_two_token_line(tmp_path):
    """``<digest>  <filename>`` (conventional sha256sum output) parses correctly."""
    manifest_path = tmp_path / "sha256.txt"
    digest = _sha256(b"hello world")
    manifest_path.write_text(f"{digest}  policy.bin\n")
    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=object(),
        hf_download=lambda **_: "",
    )
    assert poller._parse_sha256_manifest(str(manifest_path)) == digest


@pytest.mark.asyncio
async def test_poll_once_returns_false_on_empty_manifest(tmp_path):
    """End-to-end: empty manifest at HEAD causes poll_once to fail closed."""

    def _download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: str,
    ) -> str:
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        # Empty manifest — the failure mode Gemini #2 flagged.
        target.write_text("" if filename == "sha256.txt" else "")
        return str(target)

    class _FakeApi:
        def repo_info(self, repo_id: str) -> _FakeRepoInfo:
            return _FakeRepoInfo(sha="abc123")

    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=_FakeApi(),
        hf_download=_download,
    )
    result = await poller.poll_once()
    assert result is False
    # Empty manifest is fail-closed: do NOT mark the revision as seen so a
    # corrected upstream upload can be re-attempted on the next cycle.
    assert poller._last_known_sha is None


# ---------------------------------------------------------------------------
# Download timeout + retry / backoff (Copilot 3253293659/3253309998)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_retries_on_transient_failure(tmp_path):
    """Transient network failure → retries up to max_retries → succeeds on later attempt."""
    attempts: list[int] = []

    def _flaky_download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: str,
    ) -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("simulated transient network failure")
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        target.write_text("ok")
        return str(target)

    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=object(),
        hf_download=_flaky_download,
        max_retries=3,
    )
    # Backoff is 2**attempt seconds — patch the poller module's sleep
    # reference so the test doesn't actually wait wall-clock seconds.
    # Capture the real asyncio.sleep BEFORE patching to avoid recursion.
    import mousedroid.cloud.weight_update_poller as poller_module

    real_sleep = asyncio.sleep

    async def _noop_sleep(_t: float) -> None:
        await real_sleep(0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(poller_module.asyncio, "sleep", _noop_sleep)
        result = await poller._download_with_timeout_and_retry(
            _flaky_download,
            filename="policy.bin",
            revision="abc123",
        )
    assert result.endswith("policy.bin")
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_download_raises_runtime_error_after_exhausting_retries(tmp_path):
    """All attempts exhausted → RuntimeError so caller can fail closed cleanly."""

    def _always_fails(**_: Any) -> str:
        raise RuntimeError("permanent failure")

    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=object(),
        hf_download=_always_fails,
        max_retries=1,  # 2 attempts total
    )
    import mousedroid.cloud.weight_update_poller as poller_module

    real_sleep = asyncio.sleep

    async def _noop_sleep(_t: float) -> None:
        await real_sleep(0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(poller_module.asyncio, "sleep", _noop_sleep)
        with pytest.raises(RuntimeError):
            await poller._download_with_timeout_and_retry(
                _always_fails,
                filename="policy.bin",
                revision="abc123",
            )


@pytest.mark.asyncio
async def test_download_raises_timeout_when_exhausted(tmp_path):
    """Timeout on every attempt → asyncio.TimeoutError reaches caller.

    Uses a blocking ``time.sleep`` inside the to_thread worker so the
    real wall-clock timer (which ``asyncio.wait_for`` honours) actually
    fires. The retry-backoff ``asyncio.sleep`` is patched to a no-op so
    the test completes quickly between attempts.
    """
    import time

    def _hanging_download(**_kwargs: Any) -> str:
        # Blocking sleep in a worker thread — wait_for cancels after timeout.
        time.sleep(1.0)
        return ""

    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=object(),
        hf_download=_hanging_download,
        download_timeout_s=0.05,
        max_retries=1,
    )
    import mousedroid.cloud.weight_update_poller as poller_module

    real_sleep = asyncio.sleep

    async def _noop_sleep(_t: float) -> None:
        await real_sleep(0)

    with pytest.MonkeyPatch.context() as mp:
        # Patch only the retry-backoff sleep, NOT to_thread — the worker
        # thread needs the real timer for wait_for to cancel it.
        mp.setattr(poller_module.asyncio, "sleep", _noop_sleep)
        with pytest.raises(asyncio.TimeoutError):
            await poller._download_with_timeout_and_retry(
                _hanging_download,
                filename="policy.bin",
                revision="abc123",
            )


# ---------------------------------------------------------------------------
# Protected-path validation on cache_dir (Copilot 3253293703/3253310034)
# ---------------------------------------------------------------------------


def test_constructor_rejects_protected_cache_dir(monkeypatch, tmp_path):
    """Construction MUST fail fast for a cache_dir under a protected root.

    Reuses the same allowlist weights_manager.download_weights_from_huggingface
    applies — operators can't accidentally configure the poller to write
    into /etc, /root, /sys, etc. Patches the protected-root list with a
    Windows-portable path under tmp_path so this test runs the same way
    on every CI runner regardless of OS-specific path semantics.
    """
    from mousedroid.utils import weights_manager

    fake_protected = (tmp_path / "fake_etc").resolve()
    fake_protected.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        weights_manager,
        "_PROTECTED_DOWNLOAD_ROOTS",
        (fake_protected,),
    )

    cfg = WeightUpdatePollConfig(
        poll_interval_s=30.0,
        policy_repo_id="ianshank/test",
        policy_filename="policy.bin",
        cache_dir=str(fake_protected / "weights"),
        sha256_manifest_filename="sha256.txt",
    )
    with pytest.raises(ValueError, match="refusing to write"):
        HuggingFaceWeightUpdatePoller(
            cfg,
            repo_id=cfg.policy_repo_id,
            filename=cfg.policy_filename,
            engine_type="policy",
        )


@pytest.mark.asyncio
async def test_run_loop_swallows_poll_failures_and_continues(tmp_path):
    """The background ``_run`` loop catches per-cycle exceptions and continues.

    Regression net for the broad-except in ``_run`` — without this test
    the loop's except-and-log branch is uncovered, and a future refactor
    that removes the catch (or narrows it) could turn a single transient
    HF Hub error into a poller-task death silently halting all OTA
    updates.
    """
    call_count = 0

    class _FlakyApi:
        def repo_info(self, repo_id: str) -> _FakeRepoInfo:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated transient HF Hub failure")
            return _FakeRepoInfo(sha="abc123")

    def _download(**_kwargs: Any) -> str:
        target_dir = Path(_kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / _kwargs["filename"]
        digest = _sha256(b"artifact")
        if _kwargs["filename"] == "sha256.txt":
            target.write_text(digest)
        else:
            target.write_bytes(b"artifact")
        return str(target)

    # Use very short poll_interval so two cycles complete quickly.
    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=_FlakyApi(),
        hf_download=_download,
        poll_interval_s=0.01,
    )
    await poller.start()
    # Give the loop time to run a few cycles.
    await asyncio.sleep(0.05)
    await poller.stop()
    # First cycle errored + got logged; subsequent cycles succeeded so the
    # pending slot now holds the verified update.
    assert call_count >= 2


@pytest.mark.asyncio
async def test_poll_once_returns_false_on_artifact_download_failure(tmp_path):
    """Artifact download failure (after manifest OK) must fail closed.

    Covers the second fail-closed path: manifest downloads fine, but the
    main artifact download exhausts retries → poller logs the
    ``cloud_weight_update_artifact_download_failed`` event and returns False.
    """

    def _download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: str,
    ) -> str:
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if filename == "sha256.txt":
            target = target_dir / filename
            target.write_text(_sha256(b"placeholder"))
            return str(target)
        # Artifact download fails permanently.
        raise RuntimeError("simulated permanent artifact failure")

    class _FakeApi:
        def repo_info(self, repo_id: str) -> _FakeRepoInfo:
            return _FakeRepoInfo(sha="abc999")

    poller = _build_poller_with_overrides(
        tmp_path,
        hf_api=_FakeApi(),
        hf_download=_download,
        max_retries=0,  # 1 attempt, no retries — keep test fast
    )
    import mousedroid.cloud.weight_update_poller as poller_module

    real_sleep = asyncio.sleep

    async def _noop_sleep(_t: float) -> None:
        await real_sleep(0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(poller_module.asyncio, "sleep", _noop_sleep)
        result = await poller.poll_once()
    assert result is False
    # Revision NOT recorded so the corrected artifact can be re-attempted.
    assert poller._last_known_sha is None
    assert poller._pending_update is None


def test_start_idempotent_when_already_running(tmp_path):
    """Calling start() twice must not spawn a second background task.

    Covers the ``if self._task is not None and not self._task.done():``
    early-return branch (lines 123-124).
    """
    import asyncio

    async def _run() -> None:
        cfg = _make_cfg(tmp_path, poll_interval_s=10.0)
        poller = HuggingFaceWeightUpdatePoller(
            cfg,
            repo_id=cfg.policy_repo_id,
            filename=cfg.policy_filename,
            engine_type="policy",
            hf_api_factory=lambda: object(),
            hf_download=lambda **_: "",
        )
        await poller.start()
        first_task = poller._task
        await poller.start()  # idempotent
        assert poller._task is first_task
        await poller.stop()

    asyncio.run(_run())
