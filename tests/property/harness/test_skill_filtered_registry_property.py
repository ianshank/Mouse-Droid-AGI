"""Property tests for ``FilteredToolRegistry``."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from mousedroid.common.tools.registry import ToolRegistry, ToolSpec
from mousedroid.skills.registry import FilteredToolRegistry


async def _h() -> str:
    return "ok"


_tool_names = st.text(
    alphabet="abcdefghij",
    min_size=1,
    max_size=4,
)


@given(
    parent_names=st.lists(_tool_names, min_size=0, max_size=10, unique=True),
    allowed=st.lists(_tool_names, min_size=0, max_size=10, unique=True),
)
@settings(max_examples=80, deadline=None)
def test_filtered_registry_subset_invariant(parent_names: list[str], allowed: list[str]) -> None:
    parent = ToolRegistry()
    for n in parent_names:
        parent.register(ToolSpec(name=n, description="", handler=_h))

    filtered = FilteredToolRegistry(parent, allowed)
    # Names exposed must be a subset of the parent registry AND the
    # ``allowed`` whitelist (they must satisfy both).
    exposed = set(filtered.names)
    assert exposed.issubset(set(parent_names))
    assert exposed.issubset(set(allowed))
    # Every parent tool that is NOT in ``allowed`` must be invisible.
    for n in parent_names:
        if n not in allowed:
            assert filtered.get(n) is None
