"""Property-based tests for the MCP module.

Two invariants we rely on:

* Every name returned by ``MCPToolBridge.visible_tool_names()`` MUST be
  dispatchable via ``call_tool`` (modulo gates the property explicitly
  reasons about), regardless of the deny/allow/actuation choices.
* The redaction helper MUST never let a key matching the configured
  pattern survive in the output, no matter how the input is nested.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.config.schema import MCPConfig, Settings
from mousedroid.mcp.resources import REDACTED, redact_value
from mousedroid.mcp.tool_bridge import MCPToolBridge

REDACT_PATTERN = re.compile(MCPConfig.model_fields["redact_key_pattern"].default)


def _bridge_with_tools(tool_names: list[str], denylist: list[str]) -> MCPToolBridge:
    reg = ToolRegistry()

    async def _ok() -> dict[str, str]:
        return {"status": "ok"}

    for name in tool_names:
        reg.register(ToolSpec(name, "synthetic", _ok))

    cfg = MCPConfig.model_validate(
        {
            "enabled": True,
            "tools_denylist": [n for n in denylist if n != "health_check"],
        }
    )
    monitor = MagicMock()
    monitor.evaluate.return_value = MagicMock(is_emergency=False)
    return MCPToolBridge(
        cfg=cfg,
        root_cfg=Settings.model_validate({"mock_hardware": True}),
        tool_registry=reg,
        safety_monitor=monitor,
    )


_VALID_NAME = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=2,
    max_size=12,
).filter(lambda n: n != "health_check")


@settings(max_examples=30, deadline=None)
@given(
    names=st.lists(_VALID_NAME, min_size=1, max_size=6, unique=True),
    deny_size=st.integers(min_value=0, max_value=3),
)
def test_every_visible_tool_is_dispatchable(names: list[str], deny_size: int) -> None:
    """visible_tool_names() agrees with what call_tool actually accepts."""
    deny = list(names[:deny_size])
    bridge = _bridge_with_tools([*names, "health_check"], deny)
    visible = bridge.visible_tool_names()
    # Property: nothing in deny is visible
    for d in deny:
        assert d not in visible
    # Property: every visible name dispatches successfully
    ctx = bridge.make_request_context()
    for name in visible:
        result = asyncio.run(bridge.call_tool(name, None, ctx))
        # The non-actuation default tools can either succeed or hit the
        # rate limiter — but they must NEVER come back denied / unknown.
        assert result.status in {
            "ok",
            "rate_limited",
        }, f"{name} unexpectedly returned {result.status}"


@settings(max_examples=50, deadline=None)
@given(
    payload=st.recursive(
        st.one_of(
            st.text(max_size=10),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=children,
                max_size=4,
            ),
        ),
        max_leaves=12,
    ),
)
def test_redact_never_leaks_secret_key(payload) -> None:
    """No key matching the secret pattern survives in the output."""
    out = redact_value(payload, key_pattern=REDACT_PATTERN)
    _walk_and_assert_redacted(out)


def _walk_and_assert_redacted(value) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and REDACT_PATTERN.search(k):
                # Must be the sentinel — never the original value.
                assert v == REDACTED
            else:
                _walk_and_assert_redacted(v)
    elif isinstance(value, list):
        for item in value:
            _walk_and_assert_redacted(item)
