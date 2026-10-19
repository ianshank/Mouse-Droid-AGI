"""A latched rover's boot hint inspects it; clearing is a separate, typed act (F-059 4.7, R44).

The whole defect was an operator following the documented path and getting a clean-looking exit 0
that cleared nothing -- and the fix must not swing to the opposite hazard, a logged command that
clears for whoever pastes it. So the chain is walked end to end, with no help from the environment:

1. a rover started with ``--config <overlay>`` trips and persists the latch;
2. the next boot logs ``rearm_cmd``;
3. the *tokens of that very string* are fed to the CLI: it reports LATCHED, exits 1 and **changes
   nothing** (pasting the hint is a look, never a clear);
4. only then does the operator type the clear form themselves -- ``--operator`` and
   ``--confirm-area-clear`` are theirs to supply -- and the CLI clears it, exit 0;
5. a third boot comes up unlatched.

The four config env vars are removed first, so the only thing that can have told the CLI where
to look is the hint.
"""

from __future__ import annotations

import asyncio
import io
import shlex
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import mousedroid.cli.rearm as rearm_mod
import mousedroid.orchestrator._lifecycle_mixin as lifecycle_mod
from mousedroid.cli.rearm import EXIT_OK, EXIT_STILL_LATCHED, main
from mousedroid.config.loader import load_settings
from mousedroid.validation.runtime import RUNTIME_CONFIG_ENV_VARS

_OPERATOR = "e2e-operator"
_REASON = "forward_clearance_violation"


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, _level: str) -> Any:
        def _record(event: str, **kw: Any) -> None:
            self.events.append((event, kw))

        return _record


class _Streams:
    """What the CLI itself wrote, free of the structlog lines other modules print."""

    def __init__(self) -> None:
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()


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


async def test_pasting_the_boot_hint_inspects_and_clearing_stays_an_operator_act(
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
    tripped.trip(_REASON, causes=(_REASON,))
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
    events = [kw for name, kw in log.events if name == "estop_latch_active_at_boot"]
    assert len(events) == 1, f"a latched boot must log the hint exactly once: {log.events}"
    hint = events[0]["rearm_cmd"]
    assert "--confirm-area-clear" not in hint, "the logged command must not pre-attest (Ruling 44)"

    # --- step 3: the operator pastes the hint. It looks; it does not clear. ------------
    streams = _Streams()
    monkeypatch.setattr(rearm_mod, "sys", streams)
    argv = shlex.split(hint)
    assert argv[:3] == ["python", "-m", "mousedroid.cli.rearm"]
    pasted = argv[3:]
    # ``main`` runs its own event loop, so it cannot be called from this running one.
    status_code = await asyncio.to_thread(main, pasted)

    out = streams.stdout.getvalue()
    assert status_code == EXIT_STILL_LATCHED
    assert "EMERGENCY STOP LATCHED" in out
    assert _REASON in out
    assert latch_file.exists(), "pasting the logged hint cleared the latch: it must only inspect"

    # --- step 4: after looking at the rover, the operator TYPES the clear form -----------
    # (the ``--config`` tokens are the hint's; ``--operator`` and ``--confirm-area-clear`` are
    # supplied by the human, which is the point.)
    config_tokens = pasted[: pasted.index("--status")]
    streams.stdout.seek(0)
    streams.stdout.truncate()
    clear_code = await asyncio.to_thread(
        main, [*config_tokens, "--operator", _OPERATOR, "--confirm-area-clear"]
    )

    assert clear_code == EXIT_OK
    assert f"Cleared by {_OPERATOR}" in streams.stdout.getvalue()
    assert not latch_file.exists(), "the typed clear form did not clear the latch"

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
