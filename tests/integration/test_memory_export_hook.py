"""Integration test for the orchestrator's MEMORY.md export hook (Phase D).

Drives ``MouseDroidOrchestrator._maybe_export_memory`` directly so the
test stays fast and does not need to spin up the 30 Hz tick loop. The
goal is to prove the gating logic — the exporter only runs when the
mission dispatcher has flagged a completed mission AND the tick count
is on the configured cadence.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mousedroid.config.schema import (
    MemoryConfig,
    OpenClawConfig,
    Settings,
)
from mousedroid.memory.episodic import EpisodicReplay
from mousedroid.memory.exporter import MarkdownReplayExporter
from mousedroid.orchestrator.mission_dispatcher import (
    DeferredOrchestratorRef,
    OrchestratorMissionDispatcher,
)
from mousedroid.security.injection_filter import RegexInjectionFilter


def _replay() -> EpisodicReplay:
    cfg = MemoryConfig(episodic_capacity=8)
    r = EpisodicReplay(cfg, seed=0)
    for i in range(4):
        r.push({"step": i})
    return r


def _dispatcher() -> OrchestratorMissionDispatcher:
    deferred = DeferredOrchestratorRef()
    deferred.bind(MagicMock())  # never actually called in these tests
    return OrchestratorMissionDispatcher(
        deferred,
        injection_filter=RegexInjectionFilter([], max_len=64),
        cfg=OpenClawConfig(enabled=True, export_every_n_ticks=5),
    )


def _orchestrator(
    *,
    exporter: MarkdownReplayExporter | None,
    dispatcher: OrchestratorMissionDispatcher | None,
    tier: object | None,
    cfg: Settings | None = None,
) -> object:
    """Build a minimal stand-in that exposes the export-hook surface.

    Importing the full ``MouseDroidOrchestrator`` would drag in the
    entire factory; we only need to exercise ``_maybe_export_memory``
    so a tiny shim suffices and keeps the test focused on the hook.
    """

    from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator

    s = MouseDroidOrchestrator.__new__(MouseDroidOrchestrator)
    s._memory_exporter = exporter  # type: ignore[attr-defined]
    s._memory_tier = tier  # type: ignore[attr-defined]
    s._mission_dispatcher = dispatcher  # type: ignore[attr-defined]
    s._memory_export_every_n = (
        cfg.openclaw.export_every_n_ticks if cfg is not None and cfg.openclaw is not None else 5
    )  # type: ignore[attr-defined]
    s._tick_count = 0  # type: ignore[attr-defined]
    return s


@pytest.mark.asyncio
async def test_no_op_when_exporter_missing(tmp_path: Path) -> None:
    s = _orchestrator(exporter=None, dispatcher=_dispatcher(), tier=MagicMock(episodic=_replay()))
    s._tick_count = 5  # type: ignore[attr-defined]
    s._mission_dispatcher._mission_completed = True  # type: ignore[attr-defined]
    await s._maybe_export_memory()  # type: ignore[attr-defined]
    assert not any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_no_op_when_dispatcher_missing(tmp_path: Path) -> None:
    exporter = MarkdownReplayExporter(tmp_path / "MEMORY.md")
    s = _orchestrator(exporter=exporter, dispatcher=None, tier=MagicMock(episodic=_replay()))
    s._tick_count = 5  # type: ignore[attr-defined]
    await s._maybe_export_memory()  # type: ignore[attr-defined]
    assert not (tmp_path / "MEMORY.md").exists()


@pytest.mark.asyncio
async def test_no_op_when_mission_not_completed(tmp_path: Path) -> None:
    exporter = MarkdownReplayExporter(tmp_path / "MEMORY.md")
    dispatcher = _dispatcher()
    # mission_just_completed defaults to False
    s = _orchestrator(exporter=exporter, dispatcher=dispatcher, tier=MagicMock(episodic=_replay()))
    s._tick_count = 5  # type: ignore[attr-defined]
    await s._maybe_export_memory()  # type: ignore[attr-defined]
    assert not (tmp_path / "MEMORY.md").exists()


@pytest.mark.asyncio
async def test_no_op_when_off_cadence(tmp_path: Path) -> None:
    exporter = MarkdownReplayExporter(tmp_path / "MEMORY.md")
    dispatcher = _dispatcher()
    dispatcher._mission_completed = True  # type: ignore[attr-defined]
    s = _orchestrator(exporter=exporter, dispatcher=dispatcher, tier=MagicMock(episodic=_replay()))
    s._tick_count = 7  # not divisible by 5  # type: ignore[attr-defined]
    await s._maybe_export_memory()  # type: ignore[attr-defined]
    assert not (tmp_path / "MEMORY.md").exists()
    # Flag stays set since the hook didn't run.
    assert dispatcher.mission_just_completed is True


@pytest.mark.asyncio
async def test_exports_when_all_gates_pass_and_clears_flag(tmp_path: Path) -> None:
    out_path = tmp_path / "MEMORY.md"
    exporter = MarkdownReplayExporter(out_path)
    dispatcher = _dispatcher()
    dispatcher._mission_completed = True  # type: ignore[attr-defined]
    s = _orchestrator(exporter=exporter, dispatcher=dispatcher, tier=MagicMock(episodic=_replay()))
    s._tick_count = 5  # divisible by export_every_n_ticks  # type: ignore[attr-defined]
    await s._maybe_export_memory()  # type: ignore[attr-defined]
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert "## Recent experiences" in body
    # Hook clears the flag so the next mission gets exactly one export.
    assert dispatcher.mission_just_completed is False


@pytest.mark.asyncio
async def test_exporter_failure_does_not_crash_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken exporter must not propagate exceptions to the control loop."""

    class _BrokenExporter:
        async def export(self, _replay: EpisodicReplay) -> Path | None:
            raise RuntimeError("disk on fire")

    dispatcher = _dispatcher()
    dispatcher._mission_completed = True  # type: ignore[attr-defined]
    s = _orchestrator(
        exporter=_BrokenExporter(),  # type: ignore[arg-type]
        dispatcher=dispatcher,
        tier=MagicMock(episodic=_replay()),
    )
    s._tick_count = 5  # type: ignore[attr-defined]
    # Must not raise.
    await s._maybe_export_memory()  # type: ignore[attr-defined]
    # Flag is cleared in the finally block even on failure.
    assert dispatcher.mission_just_completed is False
