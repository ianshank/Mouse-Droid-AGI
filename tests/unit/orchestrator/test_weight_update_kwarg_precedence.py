"""Tier C1.2 back-compat precedence: explicit mapping wins over legacy kwarg.

The orchestrator constructor accepts both the legacy single
``weight_update_poller=`` kwarg and the C1.2 ``weight_update_pollers=``
mapping kwarg. Precedence is keyed on whether the *mapping kwarg was
provided* (i.e. not ``None``), not on whether it is non-empty — so an
explicit ``weight_update_pollers={}`` cleanly disables OTA without being
silently overridden by an accidentally-passed legacy poller.

These tests pin the new precise condition introduced in this hardening pass
(fixes the prior ``not self._weight_update_pollers`` check that would have
silently merged an empty mapping with the legacy single poller).
"""

from __future__ import annotations

from mousedroid.config.schema import Settings
from tests.unit.orchestrator.test_weight_update_swap import _build_orch, _StubPoller


def test_legacy_kwarg_used_when_mapping_kwarg_omitted() -> None:
    """``weight_update_pollers=None`` → legacy kwarg is honoured (back-compat)."""
    cfg = Settings(mock_hardware=True)
    legacy_poller = _StubPoller([])
    orch, _, _ = _build_orch(cfg, poller=legacy_poller)
    # Legacy poller folded into the internal mapping under "policy".
    assert "policy" in orch._weight_update_pollers
    assert orch._weight_update_pollers["policy"] is legacy_poller


def test_empty_mapping_disables_ota_and_ignores_legacy_kwarg(capsys) -> None:
    """``weight_update_pollers={}`` + legacy kwarg → mapping wins (empty), warning logged.

    Regression net for the imprecise condition that previously folded a
    legacy poller in whenever ``self._weight_update_pollers`` was falsy —
    an explicit empty mapping is a deliberate "disable OTA" signal and
    must NOT be silently overridden by a stale legacy kwarg.
    """
    cfg = Settings(mock_hardware=True)
    legacy_poller = _StubPoller([])
    orch, _, _ = _build_orch(
        cfg,
        poller=legacy_poller,
        weight_update_pollers={},
    )
    assert orch._weight_update_pollers == {}
    captured = capsys.readouterr()
    assert "weight_update_poller_kwarg_ignored" in (captured.out + captured.err)


def test_nonempty_mapping_supersedes_legacy_kwarg(capsys) -> None:
    """Non-empty mapping wins over legacy kwarg and emits a structured warning."""
    cfg = Settings(mock_hardware=True)
    legacy_poller = _StubPoller([])
    mapping_poller = _StubPoller([])
    orch, _, _ = _build_orch(
        cfg,
        poller=legacy_poller,
        weight_update_pollers={"policy": mapping_poller},
    )
    # Mapping poller is the one stored, NOT the legacy kwarg.
    assert orch._weight_update_pollers["policy"] is mapping_poller
    captured = capsys.readouterr()
    assert "weight_update_poller_kwarg_ignored" in (captured.out + captured.err)


def test_legacy_kwarg_with_engine_type_property_routed_to_correct_slot() -> None:
    """A legacy poller exposing ``engine_type`` lands in the right map slot.

    Verifies the new ``engine_type`` Protocol property is honoured over the
    deprecated private ``_engine_type`` attribute fallback chain.
    """
    cfg = Settings(mock_hardware=True)

    class _PollerWithEngineTypeProperty(_StubPoller):
        @property
        def engine_type(self) -> str:
            return "world_model"

    poller = _PollerWithEngineTypeProperty([])
    orch, _, _ = _build_orch(cfg, poller=poller)
    assert "world_model" in orch._weight_update_pollers
    assert "policy" not in orch._weight_update_pollers


def test_legacy_kwarg_with_private_attr_routed_via_fallback() -> None:
    """A poller exposing only the deprecated private ``_engine_type`` still routes.

    Documents the legacy fallback chain
    (``engine_type`` property → ``_engine_type`` private attr → ``"policy"``)
    so external pollers predating the protocol addition keep working.
    """
    cfg = Settings(mock_hardware=True)
    poller = _StubPoller([])
    poller._engine_type = "world_model"  # type: ignore[attr-defined]
    orch, _, _ = _build_orch(cfg, poller=poller)
    assert "world_model" in orch._weight_update_pollers
    assert "policy" not in orch._weight_update_pollers
