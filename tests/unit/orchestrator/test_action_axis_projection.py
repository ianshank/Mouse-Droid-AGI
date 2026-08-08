"""F-025: the executed action must equal the recorded action.

A skid-steer command set (``waveshare_stock``) has no lateral axis, so the
driver silently drops ``vy``. The orchestrator therefore projects the action
onto the executable axes *before* dispatch, so ``ctx.executed_action`` and the
experience log describe the motion that really happened.

Without this the world model — and anything replay-trained from it — is fit on
a physically inert ``action[1]``, and the discrepancy is invisible downstream
because an encoder-less chassis reports no lateral motion to contradict it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext


def _orchestrator(cfg: Settings) -> MouseDroidOrchestrator:
    """Minimal orchestrator — only the projection helper is exercised."""
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    return MouseDroidOrchestrator(
        world_model=MagicMock(),
        agents=[MagicMock()],
        safety_monitor=safety_monitor,
        esp32=AsyncMock(),
        sensor_manager=AsyncMock(),
        cfg=cfg,
    )


def _stock_cfg() -> Settings:
    return Settings(mock_hardware=True, esp32={"command_set": "waveshare_stock"})


class TestLegacyPathUnchanged:
    """The default command set executes every axis, so nothing is projected."""

    def test_action_is_returned_by_identity_not_copied(self) -> None:
        cfg = Settings(mock_hardware=True)
        assert cfg.esp32.command_set == "legacy"
        orch = _orchestrator(cfg)
        action = torch.tensor([0.5, -0.4, 0.3])

        projected = orch._project_action_to_executable_axes(action)

        # Identity, not just equality: the legacy path must stay
        # allocation-identical on the 30 Hz tick.
        assert projected is action


class TestStockChassisProjection:
    """Stock has no lateral axis; ``vy`` is zeroed before dispatch."""

    def test_lateral_component_is_zeroed(self) -> None:
        orch = _orchestrator(_stock_cfg())
        action = torch.tensor([0.5, -0.4, 0.3])

        projected = orch._project_action_to_executable_axes(action)

        assert float(projected[1]) == 0.0
        # The executable axes survive untouched.
        assert float(projected[0]) == pytest.approx(0.5)
        assert float(projected[2]) == pytest.approx(0.3)

    def test_the_caller_s_tensor_is_not_mutated(self) -> None:
        """The projection clones — an in-place edit would corrupt the source.

        ``ctx.proposed_action`` is handed to the PRE_ACTION hooks before this
        runs, so mutating in place would retroactively rewrite what those
        hooks saw.
        """
        orch = _orchestrator(_stock_cfg())
        action = torch.tensor([0.5, -0.4, 0.3])

        orch._project_action_to_executable_axes(action)

        assert float(action[1]) == pytest.approx(-0.4)

    def test_an_already_zero_lateral_axis_skips_the_clone(self) -> None:
        """Nothing to project: return by identity rather than allocating."""
        orch = _orchestrator(_stock_cfg())
        action = torch.tensor([0.5, 0.0, 0.3])

        assert orch._project_action_to_executable_axes(action) is action

    def test_a_single_axis_action_is_returned_unchanged(self) -> None:
        """No ``action[1]`` to index — must not IndexError."""
        orch = _orchestrator(_stock_cfg())
        action = torch.tensor([0.5])

        assert orch._project_action_to_executable_axes(action) is action
