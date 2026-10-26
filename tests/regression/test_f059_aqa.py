# tests/regression/test_f059_aqa.py
"""AQA pins for F-059 (rover gate closeout).

Each test names the finding it pins: a round-3 R-id, or the PR review that raised it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from mousedroid.config.loader import load_settings
from mousedroid.config.schema import MCPConfig
from mousedroid.config.schema.hardware import WAVESHARE_STOCK_BAUD
from mousedroid.validation import runtime
from mousedroid.validation.preflight import PreflightStatus, _check_esp32
from mousedroid.validation.runtime import Esp32ProbeResult

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_PROD = _CONFIG_DIR / "jetson_production.yaml"

#: One ``MOUSEDROID_ESP32__NAME=value`` assignment, as an operator would copy it.
_ENV_ASSIGNMENT = re.compile(r"(MOUSEDROID_ESP32__[A-Z_]+)=(\w+)")


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


async def test_preflight_esp32_warns_rather_than_ok_when_the_board_is_mocked() -> None:
    """R-4 / R3-2 (S-4): the shipped production overlay mocks the ESP32.

    ``config/jetson_production.yaml`` sets ``esp32.enabled: false``, so the factory
    hands back ``MockESP32Driver``; preflight used to report ``OK driver=...`` for
    that. It must be WARN (aggregate DEGRADED) -- there is no SKIP member (R-4).
    """
    cfg = load_settings(_PROD, config_dir=_CONFIG_DIR)
    cfg.mock_hardware = False
    assert (await _check_esp32(cfg)).status is PreflightStatus.WARN


async def test_preflight_esp32_disabled_remedy_works_when_followed_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copilot review on #256: the WARN's own env remedy, applied to the shipped overlay.

    The remedy is operator-facing text, so it is executed rather than string-matched: every
    ``MOUSEDROID_ESP32__*=value`` it prints is set in the environment, the production overlay
    is reloaded, and the check must now reach the probe (stubbed -- no wire) and report OK.
    The stock baud is asserted too: the overlay pins ``serial_baud: 1000000`` and a healthy
    stock board read at that rate looks dead (F-025 vendor audit R2), so the remedy is only
    complete if the baud derives.
    """
    cfg = load_settings(_PROD, config_dir=_CONFIG_DIR)
    cfg.mock_hardware = False
    warned = await _check_esp32(cfg)
    assert warned.status is PreflightStatus.WARN

    assignments = _ENV_ASSIGNMENT.findall(warned.detail)
    assert {name for name, _ in assignments} == {
        "MOUSEDROID_ESP32__ENABLED",
        "MOUSEDROID_ESP32__COMMAND_SET",
    }, warned.detail
    for name, value in assignments:
        monkeypatch.setenv(name, value)
    followed = load_settings(_PROD, config_dir=_CONFIG_DIR)
    followed.mock_hardware = False
    assert followed.esp32.enabled is True
    assert followed.esp32.command_set == "waveshare_stock"
    assert followed.esp32.serial_baud == WAVESHARE_STOCK_BAUD

    probed: list[object] = []

    async def alive(probe_cfg: object) -> Esp32ProbeResult:
        probed.append(probe_cfg)
        return Esp32ProbeResult(True, 1001, 11.9, 0.1, "FEEDBACK_BASE_INFO after 1 poll(s)")

    monkeypatch.setattr(runtime, "collect_esp32_probe", alive)
    result = await _check_esp32(followed)
    assert result.status is PreflightStatus.OK
    assert probed == [followed]  # it reached the probe, not a config-level WARN


async def test_preflight_esp32_enabling_the_overlay_alone_only_swaps_the_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the remedy names two variables (Copilot review on #256).

    ``config/jetson_production.yaml`` leaves ``esp32.command_set`` on the schema's
    ``legacy``, so lifting ``enabled`` alone lands on the "no probe" WARN instead. Should
    the overlay ever ship ``waveshare_stock`` itself this goes red: shorten the remedy and
    delete this test.
    """
    monkeypatch.setenv("MOUSEDROID_ESP32__ENABLED", "true")
    cfg = load_settings(_PROD, config_dir=_CONFIG_DIR)
    cfg.mock_hardware = False
    result = await _check_esp32(cfg)
    assert result.status is PreflightStatus.WARN
    assert "no probe" in result.detail


def test_trend_unit_never_runs_exclusive_device_checks() -> None:
    """R-11: the on-timer preflight must never open the serial port or the camera.

    The orchestrator container owns those devices while it runs. systemd honours the LAST
    ``Environment=`` assignment of a variable, so every such line is checked, not the first.
    """
    text = (_REPO_ROOT / "scripts" / "mousedroid-trend.service").read_text(encoding="utf-8")
    declared = re.findall(r"^Environment=MOUSEDROID_TREND_CHECKS=(\S+)$", text, re.M)
    assert declared, "default check list must be an Environment= line"
    listed = {name for value in declared for name in value.split(",")}
    assert not {"esp32", "lidar", "camera"} & listed


#: Driver calls that move the rover or stop it. ``emergency_stop`` is deliberately outside
#: ``MCPConfig.actuation_tools`` so a safety emergency can never refuse it; ``esp32_diagnostics``
#: is outside that list for the opposite reason: it is a diagnostic and must stay one.
_ACTUATING_CALLS = frozenset({"send_velocity", "emergency_stop"})


def _actuator_references_by_function(path: Path) -> dict[str, set[str]]:
    """Map every function in ``path`` to the actuating names it references.

    Scans the AST, not the text: the tool's docstrings name both calls to say it never makes
    them, so a substring match would flag the explanation instead of a call. A bare reference
    (a bound method handed to a scheduler) counts, not just a call expression, and so does the
    name as a string literal (``getattr(driver, "emergency_stop")``). A docstring is one long
    literal, never exactly the bare name, so it cannot trip the literal check.
    """
    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        names |= {
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        found.setdefault(node.name, set()).update(names & _ACTUATING_CALLS)
    return found


def test_mcp_esp32_diagnostics_never_actuates() -> None:
    """D-13 / 4.3: the MCP tool reports on the board; nothing it can reach moves or stops it.

    Covers the built-in tool registry (the motor tools live in ``motor_tools.py`` by design, so
    nothing in ``registry.py`` has a reason to name an actuator) and the probe module the tool
    delegates to when it holds no driver.
    """
    modules = {
        "common/tools/registry.py": _REPO_ROOT / "src/mousedroid/common/tools/registry.py",
        "validation/runtime/_esp32.py": _REPO_ROOT / "src/mousedroid/validation/runtime/_esp32.py",
    }
    scanned = {label: _actuator_references_by_function(path) for label, path in modules.items()}
    # Guard the gate itself: a scan that never reached the handler would pass everything.
    assert "_esp32_diagnostics" in scanned["common/tools/registry.py"]
    assert "collect_esp32_probe" in scanned["validation/runtime/_esp32.py"]
    offenders = {
        f"{label}::{function}": sorted(refs)
        for label, by_function in scanned.items()
        for function, refs in by_function.items()
        if refs
    }
    assert not offenders, (
        f"esp32_diagnostics must stay read-only; these name an actuator: {offenders}"
    )


def test_esp32_diagnostics_is_not_an_actuation_tool() -> None:
    """D-13 / 4.3: ``actuation_tools`` are hidden and refused; a diagnostic must stay reachable."""
    assert "esp32_diagnostics" not in MCPConfig().actuation_tools
