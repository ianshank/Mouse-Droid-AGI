"""``OpenClawConfig.require_actuation_ack`` must actually gate actuation.

The field is documented as one half of a two-of-two interlock — "Skills
declared with metadata['actuation']=True require this flag AND
mcp.expose_actuation_tools=true. Defence-in-depth even when an operator flips
one of the two by accident" — and ``skills/builtin/navigate.py`` and the
OpenClaw SKILL.md both repeat the claim to operators.

It was read by no code. Only ``expose_actuation_tools`` was enforced, so the
documented two-of-two gate was one-of-one: an operator who enabled actuation
believing a second seatbelt was fastened had none. A documented safety gate
that does not exist is worse than an absent one, because it is relied upon.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.config.schema import Settings
from mousedroid.mcp.tool_bridge import MCPToolBridge
from mousedroid.safety.context import SafetyContext

# In ``MCPConfig.actuation_tools`` by default, so no config override is needed
# to make it actuation-classified.
_ACTUATION_TOOL = "calibrate_ultrasonic"


def _make_bridge(*, expose: bool, require_ack: bool | None) -> MCPToolBridge:
    """Build a bridge with both halves of the actuation gate set explicitly.

    ``require_ack=None`` models ``Settings.openclaw is None`` — the OpenClaw
    subsystem unwired, which is the posture of every shipped overlay.
    """
    overrides: dict[str, object] = {
        "mock_hardware": True,
        "mcp": {"enabled": True, "expose_actuation_tools": expose},
    }
    if require_ack is not None:
        # Settings rejects openclaw.enabled=true without telemetry auth (the
        # mission-dispatch endpoint must never be reachable unauthenticated),
        # so satisfy that pre-existing cross-field rule here.
        overrides["openclaw"] = {"enabled": True, "require_actuation_ack": require_ack}
        overrides["telemetry"] = {"auth": {"auth_enabled": True}}
    root_cfg = Settings.model_validate(overrides)

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    registry = ToolRegistry()
    registry.register(ToolSpec(_ACTUATION_TOOL, "actuation tool under test", _ok))

    monitor = MagicMock()
    monitor.evaluate.return_value = SafetyContext(is_emergency=False)
    return MCPToolBridge(
        cfg=root_cfg.mcp,
        root_cfg=root_cfg,
        tool_registry=registry,
        safety_monitor=monitor,
    )


class TestActuationVisibility:
    """Both flags must be true for an actuation tool to be listed."""

    def test_both_true_exposes_the_tool(self) -> None:
        bridge = _make_bridge(expose=True, require_ack=True)
        assert _ACTUATION_TOOL in bridge.visible_tool_names()

    def test_require_ack_false_hides_the_tool_even_when_exposed(self) -> None:
        """The half that was previously unenforced. Red before the fix."""
        bridge = _make_bridge(expose=True, require_ack=False)
        assert _ACTUATION_TOOL not in bridge.visible_tool_names(), (
            "require_actuation_ack=False must suppress actuation tools even "
            "when expose_actuation_tools=True — that is the whole point of a "
            "two-of-two gate."
        )

    def test_expose_false_hides_the_tool_regardless(self) -> None:
        bridge = _make_bridge(expose=False, require_ack=True)
        assert _ACTUATION_TOOL not in bridge.visible_tool_names()


async def _call(bridge: MCPToolBridge) -> str:
    """Dispatch the actuation tool and return the resulting status string."""
    result = await bridge.call_tool(_ACTUATION_TOOL, None, bridge.make_request_context())
    return result.status


class TestActuationDispatch:
    """The dispatch path must apply the same gate as the visibility filter.

    Hiding a tool from ``list_tools`` is not a gate on its own — a client that
    already knows the name can still call it — so both paths must agree.
    """

    @pytest.mark.asyncio
    async def test_require_ack_false_blocks_dispatch(self) -> None:
        status = await _call(_make_bridge(expose=True, require_ack=False))
        assert status == "actuation_disabled", (
            f"expected the actuation gate to reject dispatch, got {status!r}"
        )

    @pytest.mark.asyncio
    async def test_both_true_permits_dispatch(self) -> None:
        status = await _call(_make_bridge(expose=True, require_ack=True))
        assert status != "actuation_disabled"


class TestBackwardsCompatibility:
    """With OpenClaw unwired, behaviour must be byte-identical to before.

    Every shipped overlay has ``openclaw = None``, so this is the path that
    actually runs today. The new gate must be vacuously satisfied there rather
    than silently disabling actuation for existing deployments.
    """

    def test_openclaw_absent_leaves_expose_flag_in_sole_control(self) -> None:
        assert _ACTUATION_TOOL in _make_bridge(expose=True, require_ack=None).visible_tool_names()
        assert (
            _ACTUATION_TOOL not in _make_bridge(expose=False, require_ack=None).visible_tool_names()
        )

    @pytest.mark.asyncio
    async def test_openclaw_absent_permits_dispatch_when_exposed(self) -> None:
        status = await _call(_make_bridge(expose=True, require_ack=None))
        assert status != "actuation_disabled"
