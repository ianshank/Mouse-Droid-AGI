"""``Settings`` remembers which overlay files it was loaded from (F-059 task 4.7).

The orchestrator's ``estop_latch_active_at_boot`` hint has to name the overlay stack the
process was *started with*, and until now nothing retained it: ``main`` handed
``load_settings`` the ``--config`` paths and dropped them. This is provenance, not
configuration, so it is a pydantic private attribute behind a read-only property -- not a
schema field. That keeps every existing YAML loading unchanged (invariant 6), keeps it out
of ``model_dump``, and means a YAML file cannot forge it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import Settings


def _overlay(path: Path, doc: dict[str, object] | None = None) -> Path:
    path.write_text(yaml.safe_dump(doc or {"debug": True}), encoding="utf-8")
    return path


def test_a_directly_built_settings_has_no_overlays() -> None:
    assert Settings(mock_hardware=True).overlay_paths == ()


def test_load_settings_with_no_overlay_records_none() -> None:
    """``default.yaml`` and ``baselines.yaml`` are implicit, not overlays."""
    assert load_settings().overlay_paths == ()


def test_load_settings_records_the_overlays_in_order(tmp_path: Path) -> None:
    first = _overlay(tmp_path / "first.yaml")
    second = _overlay(tmp_path / "second.yaml", {"debug": False})

    cfg = load_settings(first, second)

    assert cfg.overlay_paths == (first.absolute(), second.absolute())


def test_a_relative_overlay_is_recorded_as_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator pastes the hint from a different cwd than the service started in."""
    _overlay(tmp_path / "rel.yaml")
    monkeypatch.chdir(tmp_path)

    cfg = load_settings(Path("rel.yaml"))

    assert cfg.overlay_paths == (Path.cwd() / "rel.yaml",)
    assert cfg.overlay_paths[0].is_absolute()


def test_the_record_survives_model_copy(tmp_path: Path) -> None:
    """``main --mock-hardware`` does ``cfg.model_copy(update=...)`` before building."""
    overlay = _overlay(tmp_path / "o.yaml")

    copied = load_settings(overlay).model_copy(update={"mock_hardware": True})

    assert copied.overlay_paths == (overlay.absolute(),)


def test_a_missing_overlay_still_raises_file_not_found(tmp_path: Path) -> None:
    """Recording provenance must not change how a bad ``--config`` fails: loudly."""
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml")


def test_it_is_provenance_not_a_schema_field() -> None:
    """Not settable from YAML or env, not dumped: it cannot be forged or leaked."""
    assert "overlay_paths" not in Settings.model_fields
    assert "overlay_paths" not in Settings(mock_hardware=True).model_dump()
    with pytest.raises(ValidationError, match="overlay_paths"):
        Settings.model_validate({"mock_hardware": True, "overlay_paths": ["/etc/x.yaml"]})


def test_the_property_is_read_only() -> None:
    with pytest.raises(AttributeError):
        Settings(mock_hardware=True).overlay_paths = ()  # type: ignore[misc]
