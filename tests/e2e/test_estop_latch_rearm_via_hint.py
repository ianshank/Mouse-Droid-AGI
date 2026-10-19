"""A latched rover's own boot hint, pasted into the CLI, clears it (F-059 task 4.7).

The whole defect was an operator following the documented path and getting a clean-looking
exit 0 that cleared nothing. So the chain is walked end to end, with no help from the
environment: a rover started with ``--config <overlay>`` trips and persists the latch; the
next boot logs ``rearm_cmd``; the *tokens of that very string* are fed to the CLI; and a
third boot comes up unlatched. The four config env vars are removed first, so the only
thing that can have told the CLI where to look is the hint.
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import mousedroid.orchestrator._lifecycle_mixin as lifecycle_mod
from mousedroid.cli.rearm import EXIT_OK, main
from mousedroid.config.loader import load_settings
from mousedroid.validation.runtime import RUNTIME_CONFIG_ENV_VARS

_OPERATOR = "e2e-operator"


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, _level: str) -> Any:
        def _record(event: str, **kw: Any) -> None:
            self.events.append((event, kw))

        return _record


def _mock_esp32() -> AsyncMock:
    esp32 = AsyncMock()
    esp32.read_encoders = AsyncMock(
        return_value=MagicMock(left_velocity_mps=0.0, right_velocity_mps=0.0, heading_rad=0.0)
    )
    return esp32


def _boot(cfg: Any) -> Any:
    """A factory-built orchestrator with the two hardware-facing drivers stood in.

    ``load_settings`` merges ``default.yaml``, which names real camera and microphone
    backends, so unlike a bare ``Settings(mock_hardware=True)`` the factory wires drivers
    that cannot start on a dev box. The latch path under test touches neither.
    """
    from mousedroid.factory import build_orchestrator

    orch = build_orchestrator(cfg)
    orch._esp32 = _mock_esp32()
    orch._sensor_manager = AsyncMock()
    return orch


async def test_the_boot_hint_is_a_working_rearm_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in RUNTIME_CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    overlay = tmp_path / "jetson_production.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "experience": {"path": str(tmp_path / "experience")},
                "safety": {"emergency_latch": {"enabled": True, "fsync": False}},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_settings(overlay)

    # --- run 1: the rover trips the latch and persists it -----------------------------
    from mousedroid.factory import build_emergency_latch

    tripped = build_emergency_latch(cfg)
    assert tripped is not None
    tripped.trip("forward_clearance_violation", causes=("forward_clearance_violation",))
    await tripped.persist()
    latch_file: Path = tripped.path  # type: ignore[attr-defined]
    assert latch_file.exists()

    # --- run 2: it boots latched and logs the hint -------------------------------------
    log = _Recorder()
    monkeypatch.setattr(lifecycle_mod, "_log", log)
    orch = _boot(cfg)
    try:
        await orch.start()
    finally:
        await orch.stop()
    hints = [kw["rearm_cmd"] for name, kw in log.events if name == "estop_latch_active_at_boot"]
    assert len(hints) == 1, f"a latched boot must log the hint exactly once: {log.events}"

    # --- the operator pastes it, substituting <you> -----------------------------------
    argv = shlex.split(hints[0])
    assert argv[:3] == ["python", "-m", "mousedroid.cli.rearm"]
    cli_args = [_OPERATOR if arg == "<you>" else arg for arg in argv[3:]]
    # ``main`` runs its own event loop, so it cannot be called from this running one.
    exit_code = await asyncio.to_thread(main, cli_args)

    assert exit_code == EXIT_OK
    assert not latch_file.exists(), "the hint's own command did not clear the latch"

    # --- run 3: the rover comes up unlatched -------------------------------------------
    log_after = _Recorder()
    monkeypatch.setattr(lifecycle_mod, "_log", log_after)
    orch_after = _boot(cfg)
    try:
        await orch_after.start()
        assert orch_after._emergency_latch is not None
        assert orch_after._emergency_latch.is_latched is False
    finally:
        await orch_after.stop()
    assert not [name for name, _ in log_after.events if name == "estop_latch_active_at_boot"]
