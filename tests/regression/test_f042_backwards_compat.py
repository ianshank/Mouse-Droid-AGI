"""F-042 backwards-compat: remaining prefixes and existing exemptions stay."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from mousedroid.config.schema import Settings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_branch_coverage.py"
_HARDCODED_PATH = _REPO_ROOT / "scripts" / "check_no_hardcoded_values.py"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_schema_telemetry_validation_prefixes_remain() -> None:
    mod = _load_module(_SCRIPT_PATH, "check_branch_coverage_f042_compat")
    assert mod._ALLOWED_DIR_PREFIXES == (
        "src/mousedroid/config/schema/",
        "src/mousedroid/telemetry/metrics/",
        "src/mousedroid/telemetry/server/",
        "src/mousedroid/validation/runtime/",
    )
    assert mod._is_exempted_from_branch_gate("src/mousedroid/config/schema/misc.py")
    assert mod._is_exempted_from_branch_gate("src/mousedroid/telemetry/metrics/registry.py")


def test_adr017_flagged_files_stay_exempt() -> None:
    """health.py and _lifecycle_mixin.py were the original gate failures."""
    mod = _load_module(_SCRIPT_PATH, "check_branch_coverage_f042_exempt")
    assert mod._is_exempted_from_branch_gate("src/mousedroid/factory/health.py")
    assert mod._is_exempted_from_branch_gate("src/mousedroid/orchestrator/_lifecycle_mixin.py")
    assert mod._is_exempted_from_branch_gate("src/mousedroid/orchestrator/_state.py")


def test_hardcoded_value_gate_still_uses_factory_prefix() -> None:
    """F-042 is the coverage gate only; the hardcoded-value script is unchanged."""
    text = _HARDCODED_PATH.read_text(encoding="utf-8")
    assert '"src/mousedroid/factory/"' in text
    assert '"src/mousedroid/orchestrator/_"' in text


@pytest.mark.parametrize(
    "overlay",
    sorted(p.name for p in (_REPO_ROOT / "config").glob("*.yaml")),
)
def test_shipped_overlays_still_load(overlay: str) -> None:
    path = _REPO_ROOT / "config" / overlay
    text = path.read_text(encoding="utf-8")
    if "# config-validator: skip" in text:
        pytest.skip(f"{overlay} carries skip marker (not a Settings overlay)")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        return
    Settings.model_validate({**raw, "mock_hardware": True})
