"""Unit tests for :class:`HuggingFaceWeightUpdatePoller` (Tier C1)."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mousedroid.cloud.protocol import (
    PendingWeightUpdate,
    WeightUpdatePollerProtocol,
)
from mousedroid.cloud.weight_update_poller import HuggingFaceWeightUpdatePoller
from mousedroid.config.schema import MetricsConfig, WeightUpdatePollConfig
from mousedroid.telemetry.metrics import MetricsRegistry


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class _FakeRepoInfo:
    sha: str


class _FakeHfApi:
    """Stub for HfApi — returns the configured revision SHA."""

    def __init__(self, sha_sequence: list[str]) -> None:
        # Pop from the front so we can simulate two cycles with different SHAs.
        self._sha_sequence = list(sha_sequence)
        self.call_count = 0

    def repo_info(self, repo_id: str) -> _FakeRepoInfo:
        self.call_count += 1
        sha = self._sha_sequence[0] if len(self._sha_sequence) == 1 else self._sha_sequence.pop(0)
        return _FakeRepoInfo(sha=sha)


def _build_hf_download(
    artifact_bytes: bytes,
    manifest_hex: str,
    *,
    artifact_filename: str,
    manifest_filename: str,
) -> Any:
    """Build a stub for ``hf_hub_download`` that writes files into local_dir."""

    def _download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: str,
    ) -> str:
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if filename == manifest_filename:
            target = target_dir / manifest_filename
            target.write_text(manifest_hex + "\n", encoding="utf-8")
        elif filename == artifact_filename:
            target = target_dir / artifact_filename
            target.write_bytes(artifact_bytes)
        else:
            raise FileNotFoundError(filename)
        return str(target)

    return _download


def _make_cfg(tmp_path: Path, *, poll_interval_s: float = 0.0) -> WeightUpdatePollConfig:
    return WeightUpdatePollConfig(
        poll_interval_s=poll_interval_s,
        policy_repo_id="ianshank/test-policy",
        policy_filename="policy.bin",
        cache_dir=str(tmp_path / "cache"),
        sha256_manifest_filename="sha256.txt",
    )


def test_poller_conforms_to_protocol(tmp_path):
    """The implementation must structurally satisfy the protocol."""
    poller = HuggingFaceWeightUpdatePoller(
        _make_cfg(tmp_path),
        repo_id="ianshank/test-policy",
        filename="policy.bin",
        engine_type="policy",
    )
    assert isinstance(poller, WeightUpdatePollerProtocol)


@pytest.mark.asyncio
async def test_poller_skips_when_revision_unchanged(tmp_path):
    """A second poll cycle with the same SHA must NOT trigger a download."""
    artifact = b"weights-v1"
    cfg = _make_cfg(tmp_path)
    api = _FakeHfApi(sha_sequence=["sha-A"])

    download_calls: list[str] = []

    def _download(**kwargs: Any) -> str:
        download_calls.append(kwargs["filename"])
        target_dir = Path(kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"] == cfg.sha256_manifest_filename:
            (target_dir / cfg.sha256_manifest_filename).write_text(_sha256(artifact))
            return str(target_dir / cfg.sha256_manifest_filename)
        (target_dir / cfg.policy_filename).write_bytes(artifact)
        return str(target_dir / cfg.policy_filename)

    poller = HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id=cfg.policy_repo_id,
        filename=cfg.policy_filename,
        engine_type="policy",
        hf_api_factory=lambda: api,
        hf_download=_download,
    )
    # First cycle downloads.
    assert await poller.poll_once() is True
    assert len(download_calls) == 2  # manifest + artifact
    # Second cycle short-circuits because the SHA is unchanged.
    download_calls.clear()
    assert await poller.poll_once() is False
    assert download_calls == []


@pytest.mark.asyncio
async def test_poller_downloads_when_revision_changes(tmp_path):
    """A new SHA on the second cycle re-downloads and surfaces a pending update."""
    artifact_v1 = b"weights-v1"
    artifact_v2 = b"weights-v2"
    cfg = _make_cfg(tmp_path)

    api = _FakeHfApi(sha_sequence=["sha-A", "sha-B"])

    current_artifact = {"bytes": artifact_v1, "hex": _sha256(artifact_v1)}

    def _download(**kwargs: Any) -> str:
        target_dir = Path(kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"] == cfg.sha256_manifest_filename:
            (target_dir / cfg.sha256_manifest_filename).write_text(current_artifact["hex"])
            return str(target_dir / cfg.sha256_manifest_filename)
        (target_dir / cfg.policy_filename).write_bytes(current_artifact["bytes"])
        return str(target_dir / cfg.policy_filename)

    poller = HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id=cfg.policy_repo_id,
        filename=cfg.policy_filename,
        engine_type="policy",
        hf_api_factory=lambda: api,
        hf_download=_download,
    )
    assert await poller.poll_once() is True
    first_update = poller.pending_update
    assert first_update is not None
    assert first_update.revision == "sha-A"

    # Operator-side ACK clears the slot so the next download is observable.
    poller.acknowledge_swap(first_update)
    assert poller.pending_update is None

    current_artifact["bytes"] = artifact_v2
    current_artifact["hex"] = _sha256(artifact_v2)
    assert await poller.poll_once() is True
    second_update = poller.pending_update
    assert second_update is not None
    assert second_update.revision == "sha-B"
    assert second_update.sha256 == _sha256(artifact_v2)


@pytest.mark.asyncio
async def test_poller_aborts_on_sha256_mismatch(tmp_path):
    """A SHA mismatch refuses the update and increments the mismatch counter."""
    artifact = b"weights-v1"
    # Manifest advertises a DIFFERENT digest — must be refused.
    wrong_hex = _sha256(b"a-completely-different-blob")
    cfg = _make_cfg(tmp_path)
    api = _FakeHfApi(sha_sequence=["sha-A"])

    def _download(**kwargs: Any) -> str:
        target_dir = Path(kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"] == cfg.sha256_manifest_filename:
            (target_dir / cfg.sha256_manifest_filename).write_text(wrong_hex)
            return str(target_dir / cfg.sha256_manifest_filename)
        (target_dir / cfg.policy_filename).write_bytes(artifact)
        return str(target_dir / cfg.policy_filename)

    metrics = MetricsRegistry(MetricsConfig())
    poller = HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id=cfg.policy_repo_id,
        filename=cfg.policy_filename,
        engine_type="policy",
        metrics=metrics,
        hf_api_factory=lambda: api,
        hf_download=_download,
    )
    assert await poller.poll_once() is False
    assert poller.pending_update is None
    rendered = metrics.render_prometheus()
    assert "mousedroid_cloud_weight_update_sha256_mismatches_total" in rendered
    assert 'repo_id="ianshank/test-policy"' in rendered


@pytest.mark.asyncio
async def test_poller_emits_structured_log_per_state_transition(tmp_path, capsys):
    """Each lifecycle transition emits a distinct structured-log event."""
    artifact = b"weights-v1"
    cfg = _make_cfg(tmp_path)
    api = _FakeHfApi(sha_sequence=["sha-A"])

    def _download(**kwargs: Any) -> str:
        target_dir = Path(kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"] == cfg.sha256_manifest_filename:
            (target_dir / cfg.sha256_manifest_filename).write_text(_sha256(artifact))
            return str(target_dir / cfg.sha256_manifest_filename)
        (target_dir / cfg.policy_filename).write_bytes(artifact)
        return str(target_dir / cfg.policy_filename)

    poller = HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id=cfg.policy_repo_id,
        filename=cfg.policy_filename,
        engine_type="policy",
        hf_api_factory=lambda: api,
        hf_download=_download,
    )
    assert await poller.poll_once() is True
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    for event in (
        "cloud_weight_update_new_revision",
        "cloud_weight_update_sha256_verified",
        "cloud_weight_update_swap_pending",
    ):
        assert event in combined, f"missing event {event} in:\n{combined}"


@pytest.mark.asyncio
async def test_poller_stop_cancels_in_flight_loop(tmp_path):
    """Calling ``stop()`` while the background loop is running cancels it cleanly."""
    cfg = _make_cfg(tmp_path, poll_interval_s=0.05)
    api = _FakeHfApi(sha_sequence=["sha-A"])
    artifact = b"weights-v1"

    def _download(**kwargs: Any) -> str:
        target_dir = Path(kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"] == cfg.sha256_manifest_filename:
            (target_dir / cfg.sha256_manifest_filename).write_text(_sha256(artifact))
            return str(target_dir / cfg.sha256_manifest_filename)
        (target_dir / cfg.policy_filename).write_bytes(artifact)
        return str(target_dir / cfg.policy_filename)

    poller = HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id=cfg.policy_repo_id,
        filename=cfg.policy_filename,
        engine_type="policy",
        hf_api_factory=lambda: api,
        hf_download=_download,
    )
    await poller.start()
    # Give the loop a chance to run once.
    await asyncio.sleep(0.1)
    await poller.stop()
    # After stop the background task is None so a second stop is idempotent.
    assert poller._task is None
    await poller.stop()


@pytest.mark.asyncio
async def test_poller_pending_update_cleared_by_acknowledge_swap(tmp_path):
    """``acknowledge_swap`` only clears the matching slot."""
    cfg = _make_cfg(tmp_path)
    api = _FakeHfApi(sha_sequence=["sha-A"])
    artifact = b"weights-v1"

    def _download(**kwargs: Any) -> str:
        target_dir = Path(kwargs["local_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"] == cfg.sha256_manifest_filename:
            (target_dir / cfg.sha256_manifest_filename).write_text(_sha256(artifact))
            return str(target_dir / cfg.sha256_manifest_filename)
        (target_dir / cfg.policy_filename).write_bytes(artifact)
        return str(target_dir / cfg.policy_filename)

    poller = HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id=cfg.policy_repo_id,
        filename=cfg.policy_filename,
        engine_type="policy",
        hf_api_factory=lambda: api,
        hf_download=_download,
    )
    assert await poller.poll_once() is True
    update = poller.pending_update
    assert update is not None

    # ACK with a stale unrelated update — slot should NOT clear.
    stale = PendingWeightUpdate(
        repo_id="other",
        filename="other.bin",
        revision="x",
        sha256="0" * 64,
        local_path=tmp_path / "x",
        downloaded_at=0.0,
        engine_type="policy",
    )
    poller.acknowledge_swap(stale)
    assert poller.pending_update is update

    # ACK with the correct update clears the slot.
    poller.acknowledge_swap(update)
    assert poller.pending_update is None


@pytest.mark.asyncio
async def test_poller_start_is_no_op_when_disabled(tmp_path, capsys):
    """``poll_interval_s = 0.0`` (default) skips background loop creation."""
    cfg = _make_cfg(tmp_path, poll_interval_s=0.0)
    poller = HuggingFaceWeightUpdatePoller(
        cfg,
        repo_id=cfg.policy_repo_id,
        filename=cfg.policy_filename,
        engine_type="policy",
    )
    await poller.start()
    assert poller._task is None
    captured = capsys.readouterr()
    assert "cloud_weight_update_poller_disabled" in (captured.out + captured.err)
    # Idempotent stop on a never-started poller.
    await poller.stop()


# ---------------------------------------------------------------------------
# Factory wiring (C1.3 factory helpers)
# ---------------------------------------------------------------------------


def test_factory_build_weight_update_poller_returns_none_when_disabled():
    """Default ``poll_interval_s = 0.0`` => factory returns ``None``."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_weight_update_loader, build_weight_update_poller

    cfg = Settings(mock_hardware=True)
    assert build_weight_update_poller(cfg) is None
    assert build_weight_update_loader(cfg) is None


def test_factory_build_weight_update_poller_returns_instance_when_enabled():
    """Flipping ``poll_interval_s`` returns a configured poller."""
    from mousedroid.cloud.protocol import WeightUpdatePollerProtocol
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_weight_update_poller

    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.poll_interval_s = 30.0
    poller = build_weight_update_poller(cfg)
    assert poller is not None
    assert isinstance(poller, WeightUpdatePollerProtocol)


def test_factory_build_weight_update_loader_returns_none_when_enabled_but_no_concrete_loader():
    """When poller is enabled the loader is still ``None`` until the operator wires it."""
    from mousedroid.config.schema import Settings
    from mousedroid.factory import build_weight_update_loader

    cfg = Settings(mock_hardware=True)
    cfg.cloud.weight_update.poll_interval_s = 30.0
    # C1 ships the seam — the operator plugs in a production loader in a
    # follow-up PR. Factory returns ``None`` to keep the orchestrator's
    # swap helper as a no-op until then.
    assert build_weight_update_loader(cfg) is None
