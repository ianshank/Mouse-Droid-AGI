"""`_OrchestratorState` attribute-schema equality — ADR-017 follow-up.

`_state.py` exists to give every orchestrator mixin a single source of truth
for cross-mixin attribute types (see its own module docstring). Nothing
previously verified that source of truth stays accurate: an attribute added
to `MouseDroidOrchestrator.__init__` but never declared here is invisible to
mypy only when some mixin happens to read it (see
`test_orchestrator_mixin_surface.py` for the sibling method-surface check);
a declaration whose backing attribute is later removed from `__init__` is
invisible to everything. This test catches both directions at once with a
single exact-equality assertion instead of a one-sided subset check.

`typing.get_type_hints(_OrchestratorState)` cannot be used here: every real
type in `_state.py` is imported only under `TYPE_CHECKING`, so resolving the
string annotations raises `NameError`. Raw `_OrchestratorState.__annotations__`
sidesteps that — it returns the declared attribute names without resolving
their string type expressions, which is all an attribute-schema check needs.
"""

from __future__ import annotations

from mousedroid.orchestrator._state import _OrchestratorState
from tests.unit.orchestrator.test_orchestrator import _make_orchestrator


def test_instance_attributes_match_declared_state_schema_exactly() -> None:
    orch = _make_orchestrator()
    declared = set(_OrchestratorState.__annotations__)
    actual = set(vars(orch))
    missing_from_state = actual - declared
    stale_in_state = declared - actual
    assert not missing_from_state, (
        f"__init__ sets attribute(s) {sorted(missing_from_state)} that _state.py "
        "does not declare -- a mixin reading one of these has no mypy --strict "
        "coverage; add it to _OrchestratorState."
    )
    assert not stale_in_state, (
        f"_state.py declares attribute(s) {sorted(stale_in_state)} that __init__ "
        "no longer sets -- remove the stale declaration or restore the attribute."
    )
