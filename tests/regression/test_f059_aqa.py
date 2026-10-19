# tests/regression/test_f059_aqa.py
"""AQA pins for F-059 (rover gate closeout). Each test names the round-3 finding it pins."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import load_settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_PROD = _CONFIG_DIR / "jetson_production.yaml"
_BENCH = _CONFIG_DIR / "jetson_bench_esp32.yaml"


def _features() -> dict[str, dict]:
    raw = yaml.safe_load((_REPO_ROOT / "features.yaml").read_text(encoding="utf-8"))
    return {f["id"]: f for f in raw["features"]}


def test_f059_is_reserved_and_critical() -> None:
    f = _features()["F-059"]
    assert f["priority"] == "critical"
    assert f["tier"] == "fast"
    assert f["validation_command"] == "bash scripts/validations/F-059.sh"


def test_f059_edge_points_the_right_way() -> None:
    """R3-1: F-059 makes F-008 passable, so F-008 waits on F-059 — never the reverse."""
    feats = _features()
    assert "F-008" not in feats["F-059"]["depends_on"]
    assert "F-059" in feats["F-008"]["depends_on"]


def test_select_next_returns_f059_while_it_is_in_progress() -> None:
    """F-059 wins the in-progress pick by priority (critical beats F-050 high / F-051 medium);
    the F-050/F-051 deferral is catalog honesty (EX-3), not the lever.
    """
    catalog = _features()
    if catalog["F-059"]["status"] != "in_progress":
        pytest.skip("F-059 is no longer in progress; the edge pin is the durable guarantee")
    if any(
        f["priority"] == "critical" and f["status"] == "in_progress"
        for fid, f in catalog.items()
        if fid != "F-059"
    ):
        pytest.skip("another critical feature is in progress; the selector tie-breaks on id")
    from mousedroid.harness import spec

    feats = spec.load_features(str(_REPO_ROOT / "features.yaml"))
    sel = spec.select_next(feats)
    assert sel.feature is not None, sel
    assert sel.feature["id"] == "F-059", sel


def test_bench_overlay_flips_exactly_esp32_enabled_on_top_of_production() -> None:
    """R3-2 / EX-1: one key, stacked after production; nothing else moves."""
    prod = load_settings(_PROD, config_dir=_CONFIG_DIR)
    bench = load_settings(_PROD, _BENCH, config_dir=_CONFIG_DIR)
    assert prod.esp32.enabled is False
    assert bench.esp32.enabled is True
    assert bench.esp32.model_copy(update={"enabled": False}) == prod.esp32
    assert bench.safety == prod.safety
    assert yaml.safe_load(_BENCH.read_text(encoding="utf-8")) == {"esp32": {"enabled": True}}


def test_bench_overlay_leaves_the_motion_gate_closed() -> None:
    """Review focus 2: enabling the board never opens the Pydantic motion gate."""
    bench = load_settings(_PROD, _BENCH, config_dir=_CONFIG_DIR)
    assert bench.esp32.smoke_test_allow_motion is False
