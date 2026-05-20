"""PR #104 sanity smoke — every touched module imports + YAMLs load.

This is the *cheapest* possible regression net: if any of the modules PR #104
touched (or the YAML fixtures in the repo) silently break at import / parse
time, this file fails in under a second. CI runs it first so a typo doesn't
waste five minutes on the full suite.

Pinned surface:

* ``src/mousedroid/config/schema.py``
* ``src/mousedroid/factory.py`` (the ``build_esp32_driver`` branch)
* ``src/mousedroid/hardware/camera/jetson_csi.py`` (the ``capture_raw_jpeg``
  path)
* ``tools/dashboard_proxy.py``
* Every ``config/*.yaml`` fixture committed to the repo.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Import sanity — every touched src module loads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "mousedroid.config.schema",
        "mousedroid.factory",
        "mousedroid.hardware.camera.jetson_csi",
        "mousedroid.hardware.camera.mock_camera",
        "mousedroid.hardware.protocols",
        "mousedroid.comms.protocol",
        "mousedroid.comms.mock_driver",
        "mousedroid.resilience.resilient_driver",
    ],
)
def test_module_imports_clean(module_name: str) -> None:
    """Each module on the PR #104 surface imports without raising."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_dashboard_proxy_module_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tools/dashboard_proxy.py`` imports as a standalone module.

    The proxy isn't in ``src/``, so it's not on ``sys.path`` by default —
    spec-load it the same way the unit-test helper does. The proxy parses
    ``sys.argv`` at import time, so we neutralise pytest's argv first.
    """
    proxy_path = _REPO_ROOT / "tools" / "dashboard_proxy.py"
    assert proxy_path.exists(), f"{proxy_path} is missing"
    # The proxy parses sys.argv at import. Set a single-entry argv so the
    # env-fallback branch fires (deterministic across CI machines + locals).
    monkeypatch.setattr(sys, "argv", ["dashboard_proxy.py"])
    monkeypatch.delenv("JETSON_HTTP", raising=False)
    monkeypatch.delenv("JETSON_TOKEN", raising=False)
    monkeypatch.delenv("PROXY_PORT", raising=False)
    sys.modules.pop("dashboard_proxy", None)
    spec = importlib.util.spec_from_file_location("dashboard_proxy", proxy_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_proxy"] = mod
    spec.loader.exec_module(mod)
    # The settings resolver populates module-level constants — confirm they
    # exist + are well-typed (a refactor accidentally dropping one would
    # break `main()` at runtime, not import time, without this guard).
    assert isinstance(mod.PROXY_PORT, int)
    assert isinstance(mod.UPSTREAM_HTTP, str)
    assert isinstance(mod.UPSTREAM_WS, str)
    assert isinstance(mod._AUTH_HEADER, dict)


# ---------------------------------------------------------------------------
# YAML sanity — every fixture in config/ loads against the live schema
# ---------------------------------------------------------------------------


def _standalone_config_yaml_paths() -> list[Path]:
    """Standalone ``config/*.yaml`` fixtures — loadable without an overlay.

    The repo also ships overlay-style fragments (``jetson_*.yaml``) that
    expect to be merged on top of ``default.yaml`` before validation. The
    sanity smoke targets only the standalone roots; the regression tests
    pin the overlay merge behaviour separately.
    """
    cfg_dir = _REPO_ROOT / "config"
    if not cfg_dir.exists():
        return []
    standalone_names = {"default.yaml", "robot_arm_default.yaml"}
    return sorted(p for p in cfg_dir.glob("*.yaml") if p.name in standalone_names)


@pytest.mark.parametrize("yaml_path", _standalone_config_yaml_paths(), ids=lambda p: p.name)
def test_committed_yaml_loads_against_current_schema(yaml_path: Path) -> None:
    """Standalone ``config/*.yaml`` roots parse against the live Settings schema.

    Failing here means a top-level YAML the rover boots from has drifted
    out of step with the schema. The parametrize uses the file name as the
    test id so the failing path is obvious from CI output.
    """
    from mousedroid.config.schema import Settings

    with yaml_path.open() as fh:
        data = yaml.safe_load(fh) or {}
    Settings.model_validate(data)


def test_dev_dashboard_example_yaml_loads() -> None:
    """The PR #104 ``config/dev_dashboard.yaml.example`` parses cleanly.

    Operators copy this file to ``config/dev_dashboard.yaml`` for the live-
    Jetson dashboard. If the example drifts, the docs lie and the operator
    hits a YAML / schema error on first launch — the worst possible UX for
    a "getting started" path.
    """
    from mousedroid.config.schema import Settings

    example = _REPO_ROOT / "config" / "dev_dashboard.yaml.example"
    if not example.exists():
        pytest.skip(f"{example} not committed in this revision")
    with example.open() as fh:
        data = yaml.safe_load(fh) or {}
    Settings.model_validate(data)


# ---------------------------------------------------------------------------
# Smoke — the PR #104 fields are reachable from the loaded Settings tree
# ---------------------------------------------------------------------------


def test_pr104_fields_round_trip_through_yaml() -> None:
    """A YAML round-trip preserves the new PR #104 fields' values.

    Builds Settings → dumps to dict → reloads via YAML → asserts. Catches
    any one-way validator that silently drops the new fields during
    re-validation (a recurring footgun with Pydantic ``model_dump`` +
    ``Literal`` types).
    """
    from mousedroid.config.schema import Settings

    cfg = Settings.model_validate(
        {
            "mock_hardware": True,
            "esp32": {"enabled": False},
            "camera": {
                "v4l2_grayscale_extract": False,
                "snapshot_jpeg_quality": 75,
            },
        }
    )
    dumped = cfg.model_dump(mode="json")
    redumped = yaml.safe_dump(dumped)
    reloaded = Settings.model_validate(yaml.safe_load(redumped))
    assert reloaded.esp32.enabled is False
    assert reloaded.camera.v4l2_grayscale_extract is False
    assert reloaded.camera.snapshot_jpeg_quality == 75
