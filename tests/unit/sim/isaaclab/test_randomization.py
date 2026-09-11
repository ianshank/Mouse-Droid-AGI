"""Isaac domain-randomization translator (always-on; no isaaclab)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mousedroid.sim.isaaclab.randomization import (
    apply_isaac_domain_params,
    apply_isaac_episode_extras,
    domain_write_coverage,
)


def test_none_articulation_logs_and_returns() -> None:
    apply_isaac_domain_params(None, friction=1.0, slip=0.0, mass_kg=2.7, motor_gain=1.0)


def test_duck_typed_writer_is_called() -> None:
    seen: dict[str, float] = {}

    def _write(**kwargs: float) -> None:
        seen.update(kwargs)

    art = SimpleNamespace(apply_chassis_domain_params=_write)
    apply_isaac_domain_params(art, friction=0.9, slip=0.05, mass_kg=2.5, motor_gain=1.1)
    assert seen["friction"] == 0.9
    assert seen["mass_kg"] == 2.5


def test_physx_mass_fill_when_no_writer() -> None:
    class _Buf:
        def __init__(self) -> None:
            self.filled: float | None = None

        def fill(self, value: float) -> None:
            self.filled = value

    buf = _Buf()
    view = SimpleNamespace(get_masses=lambda: buf, set_masses=lambda _buf: None)
    art = SimpleNamespace(root_physx_view=view)
    apply_isaac_domain_params(art, friction=1.0, slip=0.0, mass_kg=3.0, motor_gain=1.0)
    assert buf.filled == 3.0


def test_extras_empty_is_safe() -> None:
    apply_isaac_episode_extras({})


def test_extras_with_fields_is_safe() -> None:
    apply_isaac_episode_extras({"uart_latency_ms": 12.0, "push_force_n": 1.5})


def test_domain_write_coverage_duck_writer_writes_all() -> None:
    wrote, skipped = domain_write_coverage(duck_writer=True, mass_written=True)
    assert wrote == ("friction", "slip", "mass_kg", "motor_gain")
    assert skipped == ()


def test_domain_write_coverage_physx_mass_is_partial() -> None:
    wrote, skipped = domain_write_coverage(duck_writer=False, mass_written=True)
    assert wrote == ("mass_kg",)
    assert skipped == ("friction", "slip", "motor_gain")


def test_domain_write_coverage_missing_handle_is_unapplied() -> None:
    wrote, skipped = domain_write_coverage(duck_writer=False, mass_written=False)
    assert wrote == ()
    assert skipped == ("friction", "slip", "mass_kg", "motor_gain")


def test_none_articulation_logs_unapplied(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def _info(event: str, **_kwargs: object) -> None:
        events.append(event)

    monkeypatch.setattr("mousedroid.sim.isaaclab.randomization._log.info", _info)
    apply_isaac_domain_params(None, friction=1.0, slip=0.0, mass_kg=2.7, motor_gain=1.0)
    assert "isaac_lab_domain_params_unapplied" in events


def test_physx_mass_logs_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def _info(event: str, **_kwargs: object) -> None:
        events.append(event)

    monkeypatch.setattr("mousedroid.sim.isaaclab.randomization._log.info", _info)

    class _Buf:
        def fill(self, value: float) -> None:
            self.filled = value

    buf = _Buf()
    view = SimpleNamespace(get_masses=lambda: buf, set_masses=lambda _buf: None)
    art = SimpleNamespace(root_physx_view=view)
    apply_isaac_domain_params(art, friction=1.0, slip=0.0, mass_kg=3.0, motor_gain=1.0)
    assert "isaac_lab_domain_params_partial" in events
