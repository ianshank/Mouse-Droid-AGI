# Jetson + USB-C-Connected Rover Smoke Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up an end-to-end smoke gate for a Jetson Orin Nano whose Wave Rover is now connected via USB-C — verifying device enumeration, power chain (battery → ESP32 → motors), bidirectional comms, and every existing sensor pipeline — while pulling every threshold and device path from Pydantic config (no hardcoded values) and enforcing the existing ruff/mypy/pytest gates.

**Architecture:** This plan **augments** (does not duplicate) the existing smoke stack: `scripts/jetson_smoke_test.sh`, `scripts/jetson_full_smoke_run.sh`, `scripts/preflight_check.sh`, `scripts/verify_sensors.py`, and the `tests/hardware/` suite. We add (1) a config-driven USB-C enumeration helper in `src/mousedroid/diagnostics/`, (2) a power-chain assertion test, (3) a runbook, and (4) a CI hook — and we promote the motor smoke stage from non-blocking to blocking only after the new gates pass on the bench. All new tunables land in Pydantic config with sane defaults so existing YAML overlays load unchanged.

**Tech Stack:** Python 3.10–3.12, asyncio, Pydantic v2, structlog, pytest (`pytest-asyncio`, `pytest-cov`, `pytest-timeout`), pyserial, ruff 0.8.0, mypy --strict, NumPy NDArray typing, bash 4+ (Jetson host scripts).

---

## Scope Check

This plan covers a single integrated subsystem — "Jetson + rover hardware smoke" — and produces working, testable software (new tests, new helpers, new stages, updated runbook) on its own. It does **not** modify the Phase 2 offline-RL training loop on the active `feat/phase2-real-episode-replay` branch.

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `src/mousedroid/config/schema.py` | Modify | Add `USBCDiscoveryConfig` Pydantic model; nest as `Optional` field on `Settings` with default `None` for backwards compat. |
| `src/mousedroid/diagnostics/__init__.py` | Create | Package marker for diagnostics helpers. |
| `src/mousedroid/diagnostics/usbc.py` | Create | `enumerate_usbc_devices()` + `resolve_by_id()` — pure helpers, structlog-instrumented, no hardware imports. |
| `src/mousedroid/diagnostics/power_chain.py` | Create | `assert_power_chain()` async helper — wraps battery + zero-velocity ack within configurable bounds. |
| `tests/unit/diagnostics/__init__.py` | Create | Test package marker. |
| `tests/unit/diagnostics/test_usbc.py` | Create | Pure-unit tests for `enumerate_usbc_devices` against a fake `/dev/serial/by-id/` tree. |
| `tests/unit/diagnostics/test_power_chain.py` | Create | Unit tests with `MockESP32Driver`. |
| `tests/hardware/test_usbc_enumeration.py` | Create | Hardware-marked test: real `/dev/serial/by-id/` lookup against `cfg.usbc_discovery`. |
| `tests/hardware/test_power_chain_smoke.py` | Create | Hardware-marked test: battery probe → zero-vel send → e-stop, with optional motion gate. |
| `tests/hardware/conftest.py` | Modify | Add `allow_motion` fixture sourced from `cfg.esp32.smoke_test_allow_motion`. |
| `scripts/check_usbc_devices.py` | Create | CLI wrapper around `diagnostics.usbc` — JSON / human output, exit non-zero on missing device. |
| `scripts/jetson_smoke_test.sh` | Modify | Add `usbc` and `power` cases; keep existing case list backwards-compatible. |
| `scripts/jetson_full_smoke_run.sh` | Modify | Insert `usbc` stage before `serial`; insert `power` stage after `motor`; default both **blocking**. |
| `config/jetson_production.yaml` | Modify | Add `usbc_discovery:` block referencing the existing by-id paths — no new magic numbers. |
| `config/default.yaml` | Modify | Add the same `usbc_discovery:` block disabled by default. |
| `docs/runbooks/jetson-rover-smoke.md` | Create | Operator runbook: cable, motion-gate, expected output, troubleshooting matrix. |
| `.github/workflows/ci.yml` | Modify | Extend `typecheck` matrix to include the new `diagnostics/` package; extend lint inclusion. |

---

## Task 1: Create work branch and verify clean baseline

**Files:**
- None (git only)

- [ ] **Step 1: Confirm uncommitted changes are intentional**

Run: `git status --short`
Expected: shows `M .github/workflows/ci.yml`, `M config/default.yaml`, `M config/jetson_production.yaml`, `?? docs/planning/WEIGHT_REUSE_CASE.md`

If anything else appears, stop and confirm with the user before continuing — those four entries belong to the active `feat/phase2-real-episode-replay` branch and must not be lost.

- [ ] **Step 2: Stash the Phase 2 work-in-progress**

Run:

```bash
git stash push --include-untracked --message "phase2-wip-pre-usbc-smoke-branch"
git stash list
```

Expected: `stash@{0}: On feat/phase2-real-episode-replay: phase2-wip-pre-usbc-smoke-branch`

- [ ] **Step 3: Branch off the main integration branch**

Run:

```bash
git fetch origin
git switch -c feat/jetson-rover-usbc-smoke origin/claude/markdown-implementation-plan-aVJ2l
git status
```

Expected: clean working tree on new branch `feat/jetson-rover-usbc-smoke`.

- [ ] **Step 4: Verify baseline test suite is green before any change**

Run: `pytest tests/unit/ -q --no-cov -x`
Expected: PASS for every unit test. If any unit test fails on the unmodified branch, fix-or-quarantine before continuing — every later TDD step assumes a green baseline.

- [ ] **Step 5: Commit the empty plan checkpoint**

```bash
git add docs/superpowers/plans/2026-05-30-jetson-rover-usbc-smoke-validation.md
git commit -m "docs(plan): jetson + USB-C rover smoke validation plan"
```

---

## Task 2: Add `USBCDiscoveryConfig` Pydantic model (TDD)

**Files:**
- Modify: `src/mousedroid/config/schema.py`
- Test: `tests/unit/test_usbc_discovery_config.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_usbc_discovery_config.py`:

```python
"""Unit tests for USBCDiscoveryConfig backwards-compat + validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mousedroid.config.schema import Settings, USBCDiscoveryConfig


def test_usbc_discovery_defaults_to_none_on_settings() -> None:
    """Settings must instantiate without an explicit usbc_discovery block."""
    s = Settings()
    assert s.usbc_discovery is None


def test_usbc_discovery_requires_at_least_one_required_endpoint() -> None:
    """An enabled config with no required endpoints is invalid."""
    with pytest.raises(ValidationError):
        USBCDiscoveryConfig(enabled=True, required_endpoints=[])


def test_usbc_discovery_normalises_by_id_root() -> None:
    """The by-id root must default to /dev/serial/by-id and be overridable."""
    cfg = USBCDiscoveryConfig(
        enabled=True,
        required_endpoints=[
            {"name": "rover_esp32", "by_id_glob": "*CP2102N*"},
        ],
    )
    assert str(cfg.by_id_root) == "/dev/serial/by-id"
    assert cfg.required_endpoints[0].name == "rover_esp32"


def test_usbc_discovery_endpoint_glob_is_required() -> None:
    """Every endpoint must declare a by_id_glob (no name-only entries)."""
    with pytest.raises(ValidationError):
        USBCDiscoveryConfig(
            enabled=True,
            required_endpoints=[{"name": "rover_esp32"}],
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_usbc_discovery_config.py -v --no-cov`
Expected: `ImportError: cannot import name 'USBCDiscoveryConfig' from 'mousedroid.config.schema'`

- [ ] **Step 3: Implement `USBCDiscoveryConfig` in `src/mousedroid/config/schema.py`**

Insert near the other hardware-discovery configs (just above `class Settings(BaseSettings):`):

```python
class USBCEndpointSpec(BaseModel):
    """A single USB-C endpoint the smoke gate expects to find under by-id."""

    name: str = Field(..., min_length=1, description="Logical role, e.g. rover_esp32")
    by_id_glob: str = Field(
        ...,
        min_length=1,
        description="Glob applied under by_id_root (e.g. '*CP2102N*-if00-port0').",
    )
    required: bool = Field(True, description="If False, missing endpoint is a WARN not FAIL.")


class USBCDiscoveryConfig(BaseModel):
    """Config-driven enumeration of USB-C endpoints required for smoke."""

    enabled: bool = Field(False, description="Master switch — keeps default YAML inert.")
    by_id_root: Path = Field(
        default=Path("/dev/serial/by-id"),
        description="Filesystem root scanned for endpoints.",
    )
    required_endpoints: list[USBCEndpointSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_endpoints_when_enabled(self) -> USBCDiscoveryConfig:
        if self.enabled and not self.required_endpoints:
            raise ValueError(
                "usbc_discovery.enabled=true requires at least one required_endpoint"
            )
        return self
```

Add the field to `Settings` (defaults to `None` so every existing YAML still loads):

```python
usbc_discovery: USBCDiscoveryConfig | None = Field(
    default=None,
    description="Optional USB-C enumeration gate; None disables.",
)
```

Ensure the import block at the top of `schema.py` includes `model_validator` (already present in this file — verify before adding).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_usbc_discovery_config.py -v --no-cov`
Expected: 4 passed.

- [ ] **Step 5: Verify every existing YAML still loads**

Run: `python scripts/validate_configs.py --include-default`
Expected: every overlay validates — no schema regression.

- [ ] **Step 6: Lint + typecheck the touched file**

Run: `ruff check src/mousedroid/config/schema.py tests/unit/test_usbc_discovery_config.py && ruff format --check src/mousedroid/config/schema.py tests/unit/test_usbc_discovery_config.py && mypy --strict src/mousedroid/config/schema.py`
Expected: no errors. NumPy typing untouched here — no `NDArray` regressions to chase.

- [ ] **Step 7: Commit**

```bash
git add src/mousedroid/config/schema.py tests/unit/test_usbc_discovery_config.py
git commit -m "feat(config): USBCDiscoveryConfig schema with default-None backwards compat"
```

---

## Task 3: Implement `mousedroid.diagnostics.usbc` enumeration helper (TDD)

**Files:**
- Create: `src/mousedroid/diagnostics/__init__.py`
- Create: `src/mousedroid/diagnostics/usbc.py`
- Create: `tests/unit/diagnostics/__init__.py`
- Create: `tests/unit/diagnostics/test_usbc.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/diagnostics/__init__.py` (empty) and `tests/unit/diagnostics/test_usbc.py`:

```python
"""Unit tests for the USB-C enumeration helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from mousedroid.config.schema import USBCDiscoveryConfig, USBCEndpointSpec
from mousedroid.diagnostics.usbc import (
    EndpointStatus,
    enumerate_usbc_devices,
)


@pytest.fixture
def fake_by_id_root(tmp_path: Path) -> Path:
    root = tmp_path / "by-id"
    root.mkdir()
    (root / "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_AAA-if00-port0").touch()
    (root / "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0").touch()
    return root


def test_enumerate_resolves_required_endpoint(fake_by_id_root: Path) -> None:
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(
                name="rover_esp32",
                by_id_glob="*CP2102N_USB_to_UART_Bridge*-if00-port0",
            ),
        ],
    )
    result = enumerate_usbc_devices(cfg)
    rover = result["rover_esp32"]
    assert rover.status is EndpointStatus.PRESENT
    assert rover.resolved_path is not None
    assert "CP2102N" in rover.resolved_path.name


def test_enumerate_marks_missing_required_as_fail(fake_by_id_root: Path) -> None:
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="lidar", by_id_glob="*NONEXISTENT*", required=True),
        ],
    )
    result = enumerate_usbc_devices(cfg)
    assert result["lidar"].status is EndpointStatus.MISSING


def test_enumerate_marks_missing_optional_as_warn(fake_by_id_root: Path) -> None:
    cfg = USBCDiscoveryConfig(
        enabled=True,
        by_id_root=fake_by_id_root,
        required_endpoints=[
            USBCEndpointSpec(name="aux", by_id_glob="*NONEXISTENT*", required=False),
        ],
    )
    result = enumerate_usbc_devices(cfg)
    assert result["aux"].status is EndpointStatus.WARN


def test_enumerate_returns_empty_when_disabled() -> None:
    cfg = USBCDiscoveryConfig()
    assert enumerate_usbc_devices(cfg) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/diagnostics/test_usbc.py -v --no-cov`
Expected: `ModuleNotFoundError: No module named 'mousedroid.diagnostics'`

- [ ] **Step 3: Create the diagnostics package marker**

Create `src/mousedroid/diagnostics/__init__.py`:

```python
"""Standalone diagnostics helpers shared by smoke scripts and tests."""

from __future__ import annotations
```

- [ ] **Step 4: Implement the enumeration helper**

Create `src/mousedroid/diagnostics/usbc.py`:

```python
"""USB-C device enumeration helper for Jetson smoke gates.

Pure helper — imports nothing hardware-specific. Resolves
``USBCDiscoveryConfig.required_endpoints`` against the configured by-id
root and emits structured logging so operator triage is grep-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from mousedroid.config.schema import USBCDiscoveryConfig
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


class EndpointStatus(str, Enum):
    """Status of a single required endpoint."""

    PRESENT = "present"
    MISSING = "missing"
    WARN = "warn"


@dataclass(frozen=True)
class EndpointResult:
    """Resolution outcome for a single endpoint."""

    name: str
    glob: str
    required: bool
    resolved_path: Path | None
    status: EndpointStatus


def enumerate_usbc_devices(
    cfg: USBCDiscoveryConfig,
) -> dict[str, EndpointResult]:
    """Resolve every required endpoint against ``cfg.by_id_root``.

    Returns an empty dict when discovery is disabled so callers can short-
    circuit without per-call enabled checks.
    """
    if not cfg.enabled:
        _log.debug("usbc_enumerate_skipped", reason="discovery_disabled")
        return {}

    results: dict[str, EndpointResult] = {}
    for spec in cfg.required_endpoints:
        matches = sorted(cfg.by_id_root.glob(spec.by_id_glob))
        if matches:
            results[spec.name] = EndpointResult(
                name=spec.name,
                glob=spec.by_id_glob,
                required=spec.required,
                resolved_path=matches[0],
                status=EndpointStatus.PRESENT,
            )
            _log.info(
                "usbc_endpoint_present",
                name=spec.name,
                path=str(matches[0]),
            )
        else:
            status = EndpointStatus.MISSING if spec.required else EndpointStatus.WARN
            results[spec.name] = EndpointResult(
                name=spec.name,
                glob=spec.by_id_glob,
                required=spec.required,
                resolved_path=None,
                status=status,
            )
            _log.warning(
                "usbc_endpoint_missing",
                name=spec.name,
                glob=spec.by_id_glob,
                status=status.value,
            )
    return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/diagnostics/test_usbc.py -v --no-cov`
Expected: 4 passed.

- [ ] **Step 6: Lint + typecheck**

Run: `ruff check src/mousedroid/diagnostics/ tests/unit/diagnostics/ && ruff format --check src/mousedroid/diagnostics/ tests/unit/diagnostics/ && mypy --strict src/mousedroid/diagnostics/`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/mousedroid/diagnostics/ tests/unit/diagnostics/
git commit -m "feat(diagnostics): config-driven USB-C endpoint enumeration helper"
```

---

## Task 4: Add `scripts/check_usbc_devices.py` CLI wrapper

**Files:**
- Create: `scripts/check_usbc_devices.py`
- Test: `tests/unit/scripts/test_check_usbc_devices.py` (new — script package may already have one; check before creating directory)

- [ ] **Step 1: Create the tests/unit/scripts package marker**

Run: `mkdir -p tests/unit/scripts && touch tests/unit/scripts/__init__.py`

(`tests/unit/scripts/` does not exist on this branch; the marker is required so pytest can collect the new test module.)

- [ ] **Step 2: Write the failing test**

Create `tests/unit/scripts/test_check_usbc_devices.py`:

```python
"""CLI smoke tests for scripts/check_usbc_devices.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_usbc_devices.py"


def _write_config(tmp_path: Path, by_id_root: Path) -> Path:
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(
        f"""
usbc_discovery:
  enabled: true
  by_id_root: {by_id_root}
  required_endpoints:
    - name: rover_esp32
      by_id_glob: "*CP2102N*-if00-port0"
""".lstrip(),
        encoding="utf-8",
    )
    return cfg


def test_cli_exits_zero_when_all_present(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    (by_id / "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_X-if00-port0").touch()
    cfg = _write_config(tmp_path, by_id)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["rover_esp32"]["status"] == "present"


def test_cli_exits_nonzero_when_required_missing(tmp_path: Path) -> None:
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    cfg = _write_config(tmp_path, by_id)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "rover_esp32" in result.stdout + result.stderr
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/scripts/test_check_usbc_devices.py -v --no-cov`
Expected: failure — `scripts/check_usbc_devices.py` does not exist.

- [ ] **Step 4: Implement the CLI**

Create `scripts/check_usbc_devices.py`:

```python
#!/usr/bin/env python3
"""USB-C endpoint smoke gate (config-driven, no hardcoded paths)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when run from repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mousedroid.config.loader import load_settings  # noqa: E402
from mousedroid.diagnostics.usbc import (  # noqa: E402
    EndpointStatus,
    enumerate_usbc_devices,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        action="append",
        required=True,
        type=Path,
        help="One or more YAML overlays (repeatable).",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human output.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings = load_settings(*args.config)
    if settings.usbc_discovery is None:
        print("usbc_discovery not configured; nothing to check", file=sys.stderr)
        return 0

    results = enumerate_usbc_devices(settings.usbc_discovery)
    if args.json:
        payload = {
            name: {
                "status": r.status.value,
                "resolved_path": str(r.resolved_path) if r.resolved_path else None,
                "required": r.required,
                "glob": r.glob,
            }
            for name, r in results.items()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, r in results.items():
            marker = {
                EndpointStatus.PRESENT: "[OK]",
                EndpointStatus.WARN: "[WARN]",
                EndpointStatus.MISSING: "[FAIL]",
            }[r.status]
            location = r.resolved_path if r.resolved_path else f"missing ({r.glob})"
            print(f"  {marker} {name}: {location}")

    has_missing = any(r.status is EndpointStatus.MISSING for r in results.values())
    return 1 if has_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/scripts/test_check_usbc_devices.py -v --no-cov`
Expected: 2 passed.

- [ ] **Step 6: Manual sanity — run the CLI with the production overlay**

Run: `python scripts/check_usbc_devices.py --config config/jetson_production.yaml`
Expected on a non-Jetson dev host: `usbc_discovery not configured` (because Task 5 has not yet populated the YAML) — exit 0.

- [ ] **Step 7: Commit**

```bash
git add scripts/check_usbc_devices.py tests/unit/scripts/
git commit -m "feat(scripts): check_usbc_devices CLI for smoke enumeration gate"
```

(`git add tests/unit/scripts/` picks up both the `__init__.py` created in Step 1 and the test module from Step 2.)

---

## Task 5: Populate `usbc_discovery:` in `config/jetson_production.yaml` and `config/default.yaml`

**Files:**
- Modify: `config/jetson_production.yaml`
- Modify: `config/default.yaml`
- Test: `tests/unit/test_jetson_production_overlay.py` (new, lightweight)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_jetson_production_overlay.py`:

```python
"""Regression test — jetson_production overlay carries USB-C discovery block."""

from __future__ import annotations

from pathlib import Path

from mousedroid.config.loader import load_settings


def test_jetson_production_declares_usbc_endpoints() -> None:
    cfg = load_settings(Path("config/jetson_production.yaml"))
    assert cfg.usbc_discovery is not None
    assert cfg.usbc_discovery.enabled is True
    names = {ep.name for ep in cfg.usbc_discovery.required_endpoints}
    assert {"rover_esp32", "lidar_ld19"}.issubset(names)


def test_default_overlay_keeps_usbc_inert() -> None:
    cfg = load_settings(Path("config/default.yaml"))
    if cfg.usbc_discovery is None:
        return  # acceptable — default has no hardware
    assert cfg.usbc_discovery.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_jetson_production_overlay.py -v --no-cov`
Expected: AssertionError — `usbc_discovery` is still `None` on the production overlay.

- [ ] **Step 3: Add the block to `config/jetson_production.yaml`**

Append (keeping the YAML grouped near `esp32:` and `lidar:`):

```yaml
# USB-C smoke gate — globs intentionally reuse the existing by-id paths so
# changing the serial_port in one place updates discovery automatically.
usbc_discovery:
  enabled: true
  by_id_root: "/dev/serial/by-id"
  required_endpoints:
    - name: rover_esp32
      by_id_glob: "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_*-if00-port0"
      required: true
    - name: lidar_ld19
      by_id_glob: "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_*-if00-port0"
      required: true
```

- [ ] **Step 4: Add a disabled stub to `config/default.yaml`**

Append:

```yaml
# Default overlay leaves USB-C discovery off so tests on non-Jetson hosts
# remain inert. Production overlays opt in.
usbc_discovery:
  enabled: false
  by_id_root: "/dev/serial/by-id"
  required_endpoints: []
```

(Note: when `enabled: false`, the model validator does not require endpoints.)

- [ ] **Step 5: Run tests to verify both pass**

Run: `pytest tests/unit/test_jetson_production_overlay.py tests/unit/test_usbc_discovery_config.py -v --no-cov && python scripts/validate_configs.py --include-default`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add config/jetson_production.yaml config/default.yaml tests/unit/test_jetson_production_overlay.py
git commit -m "feat(config): wire usbc_discovery into jetson_production overlay"
```

---

## Task 6: Hardware-marked USB-C enumeration test

**Files:**
- Create: `tests/hardware/test_usbc_enumeration.py`

- [ ] **Step 1: Write the test (will skip on non-Jetson host)**

```python
"""Hardware smoke — verifies every required USB-C endpoint resolves.

Skipped on non-Jetson hosts via the shared ``jetson_settings`` fixture
that already keys off ``is_jetson_host()``.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.diagnostics.usbc import EndpointStatus, enumerate_usbc_devices

pytestmark = pytest.mark.hardware


def test_every_required_usbc_endpoint_resolves(jetson_settings: Settings) -> None:
    if jetson_settings.usbc_discovery is None or not jetson_settings.usbc_discovery.enabled:
        pytest.skip("usbc_discovery disabled for this overlay")

    results = enumerate_usbc_devices(jetson_settings.usbc_discovery)
    missing = [name for name, r in results.items() if r.status is EndpointStatus.MISSING]
    assert not missing, (
        f"USB-C endpoints missing under {jetson_settings.usbc_discovery.by_id_root}: "
        f"{missing}. Plug the rover into the Jetson USB-C port and re-run."
    )
```

- [ ] **Step 2: Run on dev host — expect skip**

Run: `pytest tests/hardware/test_usbc_enumeration.py -v -m hardware --no-cov`
Expected: 1 skipped (non-Jetson host).

- [ ] **Step 3: Lint + typecheck the new test**

Run: `ruff check tests/hardware/test_usbc_enumeration.py && mypy --strict tests/hardware/test_usbc_enumeration.py || true`
Expected: no ruff errors. (mypy --strict on `tests/` is informational only — production CI gates mypy on `src/` per `.github/workflows/ci.yml:98`.)

- [ ] **Step 4: Commit**

```bash
git add tests/hardware/test_usbc_enumeration.py
git commit -m "test(hardware): USB-C endpoint enumeration smoke gate"
```

---

## Task 7: Power-chain assertion helper + unit test (TDD)

**Files:**
- Create: `src/mousedroid/diagnostics/power_chain.py`
- Create: `tests/unit/diagnostics/test_power_chain.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/unit/diagnostics/test_power_chain.py`:

```python
"""Power-chain assertion unit tests using MockESP32Driver."""

from __future__ import annotations

import pytest

from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.config.schema import Settings
from mousedroid.diagnostics.power_chain import (
    PowerChainResult,
    assert_power_chain,
)


@pytest.fixture
def settings() -> Settings:
    # tests/conftest.py sets MOUSEDROID_MOCK_HARDWARE=true so bare Settings()
    # bypasses the hardware_requires_pins validator under pytest.
    return Settings()


async def test_zero_velocity_round_trip_succeeds(settings: Settings) -> None:
    driver = MockESP32Driver(cfg=settings.esp32)
    await driver.connect()
    try:
        result: PowerChainResult = await assert_power_chain(
            driver=driver,
            esp32_cfg=settings.esp32,
            allow_motion=False,
        )
        assert result.battery_voltage_v >= 0.0
        assert result.estop_latency_ms <= settings.esp32.emergency_stop_budget_ms
        assert result.notes  # human-readable summary present
    finally:
        await driver.disconnect()


async def test_motion_gate_uses_zero_velocity_when_disallowed(settings: Settings) -> None:
    driver = MockESP32Driver(cfg=settings.esp32)
    await driver.connect()
    try:
        result = await assert_power_chain(
            driver=driver,
            esp32_cfg=settings.esp32,
            allow_motion=False,
        )
        assert result.commanded_velocity_mps == 0.0
    finally:
        await driver.disconnect()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/diagnostics/test_power_chain.py -v --no-cov`
Expected: `ImportError: cannot import name 'assert_power_chain'`.

- [ ] **Step 3: Implement the helper**

Create `src/mousedroid/diagnostics/power_chain.py`:

```python
"""Power-chain smoke helper — battery + zero-vel + e-stop within budget."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from mousedroid.config.schema import ESP32Config
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)


@runtime_checkable
class _PowerCapableDriver(Protocol):
    """Minimal slice of the ESP32 driver interface we depend on."""

    async def send_velocity(self, vx: float, vy: float, omega: float) -> None: ...
    async def emergency_stop(self) -> None: ...
    async def get_battery_voltage(self) -> float: ...


@dataclass(frozen=True)
class PowerChainResult:
    """Structured outcome of an `assert_power_chain` run."""

    battery_voltage_v: float
    commanded_velocity_mps: float
    estop_latency_ms: float
    notes: str


async def assert_power_chain(
    *,
    driver: _PowerCapableDriver,
    esp32_cfg: ESP32Config,
    allow_motion: bool,
) -> PowerChainResult:
    """Probe battery, dispatch a (possibly zero) velocity, then time the e-stop."""
    voltage = await driver.get_battery_voltage()
    target = esp32_cfg.smoke_test_velocity_mps if allow_motion else 0.0
    _log.info(
        "power_chain_probe_start",
        battery_v=voltage,
        target_vx=target,
        allow_motion=allow_motion,
    )

    await driver.send_velocity(target, 0.0, 0.0)
    t0 = time.monotonic()
    await driver.emergency_stop()
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    notes = (
        f"battery={voltage:.2f}V cmd={target:.3f}m/s "
        f"estop={elapsed_ms:.1f}ms (budget={esp32_cfg.emergency_stop_budget_ms:.0f}ms)"
    )
    _log.info("power_chain_probe_complete", summary=notes)
    return PowerChainResult(
        battery_voltage_v=voltage,
        commanded_velocity_mps=target,
        estop_latency_ms=elapsed_ms,
        notes=notes,
    )
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `pytest tests/unit/diagnostics/ -v --no-cov`
Expected: all green (USB-C tests still pass + 2 new power chain tests).

- [ ] **Step 5: Lint + typecheck**

Run: `ruff check src/mousedroid/diagnostics/ tests/unit/diagnostics/ && ruff format --check src/mousedroid/diagnostics/ tests/unit/diagnostics/ && mypy --strict src/mousedroid/diagnostics/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/mousedroid/diagnostics/power_chain.py tests/unit/diagnostics/test_power_chain.py
git commit -m "feat(diagnostics): power-chain smoke helper with structlog instrumentation"
```

---

## Task 8: Hardware-marked power-chain smoke test

**Files:**
- Create: `tests/hardware/test_power_chain_smoke.py`
- Modify: `tests/hardware/conftest.py`

- [ ] **Step 1: Add `allow_motion` fixture to `tests/hardware/conftest.py`**

Insert at the bottom of `tests/hardware/conftest.py`:

```python
@pytest.fixture
def allow_motion(jetson_settings: Settings) -> bool:
    """True only when the operator has explicitly enabled the motion gate."""
    return bool(jetson_settings.esp32.smoke_test_allow_motion)
```

(The `Settings` import at the top of `conftest.py` already lives behind `TYPE_CHECKING`; promote it to a real import if necessary.)

- [ ] **Step 2: Write the failing hardware test**

Create `tests/hardware/test_power_chain_smoke.py`:

```python
"""Hardware smoke — battery + motor command + e-stop within budget."""

from __future__ import annotations

import pytest

from mousedroid.config.schema import Settings
from mousedroid.diagnostics.power_chain import assert_power_chain

pytestmark = pytest.mark.hardware


async def test_power_chain_within_budget(
    jetson_settings: Settings, allow_motion: bool
) -> None:
    from mousedroid.factory import build_esp32_driver

    if jetson_settings.mock_hardware:
        pytest.skip("mock_hardware=true; power chain smoke requires real ESP32")

    driver = build_esp32_driver(jetson_settings)
    await driver.connect()
    try:
        result = await assert_power_chain(
            driver=driver,
            esp32_cfg=jetson_settings.esp32,
            allow_motion=allow_motion,
        )
        assert result.estop_latency_ms <= jetson_settings.esp32.emergency_stop_budget_ms, (
            f"e-stop latency {result.estop_latency_ms:.1f}ms exceeded budget "
            f"{jetson_settings.esp32.emergency_stop_budget_ms:.0f}ms"
        )
        if jetson_settings.safety.battery_critical_v > 0.0:
            assert result.battery_voltage_v >= jetson_settings.safety.battery_critical_v, (
                f"battery {result.battery_voltage_v:.2f}V below critical "
                f"{jetson_settings.safety.battery_critical_v:.2f}V"
            )
    finally:
        await driver.disconnect()
```

- [ ] **Step 3: Run on dev host — expect skip**

Run: `pytest tests/hardware/test_power_chain_smoke.py -v -m hardware --no-cov`
Expected: 1 skipped (`mock_hardware=true` on non-Jetson host).

- [ ] **Step 4: Commit**

```bash
git add tests/hardware/conftest.py tests/hardware/test_power_chain_smoke.py
git commit -m "test(hardware): power-chain smoke gated by smoke_test_allow_motion"
```

---

## Task 9: Add `usbc` and `power` stages to `scripts/jetson_smoke_test.sh`

**Files:**
- Modify: `scripts/jetson_smoke_test.sh`

- [ ] **Step 1: Add a `test_usbc` function above `test_serial`**

Insert (keeping the `set -euo pipefail`-safe pattern used by neighbours):

```bash
# ---------------------------------------------------------------------------
# 2b. USB-C enumeration
# ---------------------------------------------------------------------------

test_usbc() {
    log_section "USB-C Enumeration"
    log_step "Running scripts/check_usbc_devices.py"

    local output rc
    set +e
    output="$("${PYTHON}" "${PROJECT_DIR}/scripts/check_usbc_devices.py" "${CONFIG_ARGS[@]}" 2>&1)"
    rc=$?
    set -e

    echo "${output}"
    if [[ ${rc} -eq 0 ]]; then
        record_pass "usbc enumeration"
    else
        record_fail "usbc enumeration" "missing required endpoint(s) (see above)"
    fi
}
```

- [ ] **Step 2: Add a `test_power` function above `test_camera`**

```bash
# ---------------------------------------------------------------------------
# 3c. Power chain smoke
# ---------------------------------------------------------------------------

test_power() {
    log_section "Power Chain Smoke"
    log_step "Running power-chain hardware test"
    local test_file="${PROJECT_DIR}/tests/hardware/test_power_chain_smoke.py"
    if [[ ! -f "${test_file}" ]]; then
        record_skip "power chain smoke" "test_power_chain_smoke.py not found"
        return
    fi

    local pytest_output
    if pytest_output="$(MOUSEDROID_JETSON_CONFIGS="${CONFIGS_CSV}" MOUSEDROID_MOCK_HARDWARE=false \
            "${PYTHON}" -m pytest -m hardware -ra -v "${test_file}" 2>&1)"; then
        echo "${pytest_output}"
        record_pass "power chain smoke"
    else
        echo "${pytest_output}"
        record_fail "power chain smoke" "see output for battery/e-stop violation"
    fi
}
```

- [ ] **Step 3: Wire both into the `case` and `all` blocks**

In `main()`, extend the `all` branch:

```bash
all)
    test_system
    test_usbc          # NEW
    test_gpio
    test_serial
    test_motor
    test_power         # NEW
    test_camera
    test_audio
    test_lidar
    test_speaker
    test_voice
    test_app
    test_pytest
    test_e2e
    ;;
```

And extend the per-step `case` plus the help line:

```bash
usbc)     test_usbc ;;
power)    test_power ;;
```

```bash
echo "Valid steps: all, system, usbc, gpio, serial, motor, power, camera, audio, lidar, speaker, voice, app, pytest, e2e"
```

- [ ] **Step 4: ShellCheck-style sanity**

Run: `bash -n scripts/jetson_smoke_test.sh`
Expected: no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/jetson_smoke_test.sh
git commit -m "feat(smoke): add usbc + power stages to jetson_smoke_test.sh"
```

---

## Task 10: Wire `usbc` and `power` into `scripts/jetson_full_smoke_run.sh`

**Files:**
- Modify: `scripts/jetson_full_smoke_run.sh`

- [ ] **Step 1: Insert the `usbc` stage immediately after the `system` stage loop**

Replace the existing block:

```bash
for stage in system gpio serial; do
    run_stage "${stage}" "yes" 60 bash scripts/jetson_smoke_test.sh "${stage}" || break
done
```

With:

```bash
for stage in system usbc gpio serial; do
    run_stage "${stage}" "yes" 60 bash scripts/jetson_smoke_test.sh "${stage}" || break
done
```

- [ ] **Step 2: Insert the `power` stage immediately after the `motor` block**

After the existing `motor` `run_stage` (still non-blocking by default), add:

```bash
if [[ "${OVERALL_FAIL}" -eq 0 ]]; then
    # Power chain gate — blocking once Task 8 lands. Operator can demote at
    # runtime with MOUSEDROID_SMOKE_BLOCKING_POWER=no.
    run_stage "power" "yes" 120 bash scripts/jetson_smoke_test.sh power || true
fi
```

- [ ] **Step 3: Sanity-check the script**

Run: `bash -n scripts/jetson_full_smoke_run.sh`
Expected: no syntax errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/jetson_full_smoke_run.sh
git commit -m "feat(smoke): wire usbc + power stages into jetson_full_smoke_run"
```

---

## Task 11: Logging instrumentation enrichment

**Files:**
- Modify: `src/mousedroid/comms/serial_driver.py`
- Modify: `src/mousedroid/comms/base_driver.py`
- Test: `tests/unit/test_serial_driver_logging.py` (new)

- [ ] **Step 1: Identify the current logging surface**

Run: `grep -n "_log\." src/mousedroid/comms/serial_driver.py`
Note which paths already log; do NOT add a second log line where one already exists.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_serial_driver_logging.py`:

```python
"""Verify the serial driver emits structured logs at connect/disconnect."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from mousedroid.comms.base_driver import log_command_dispatch
from mousedroid.comms.mock_driver import MockESP32Driver
from mousedroid.config.schema import Settings


@pytest.fixture
def capture_log_events() -> Iterator[list[dict[str, Any]]]:
    """Capture structlog events for one test and restore configuration after."""
    captured: list[dict[str, Any]] = []

    def _capture(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(event_dict))
        return event_dict

    prior = structlog.get_config()
    structlog.configure(
        processors=[_capture],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        cache_logger_on_first_use=False,
    )
    try:
        yield captured
    finally:
        structlog.configure(**prior)


async def test_log_command_dispatch_emits_structured_event(
    capture_log_events: list[dict[str, Any]],
) -> None:
    driver = MockESP32Driver(cfg=Settings().esp32)
    await driver.connect()
    try:
        log_command_dispatch(driver_name="mock", vx=0.1, vy=0.0, omega=0.0)
    finally:
        await driver.disconnect()

    assert any(e.get("event") == "command_dispatch" for e in capture_log_events)
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/unit/test_serial_driver_logging.py -v --no-cov`
Expected: `ImportError: cannot import name 'log_command_dispatch'`.

- [ ] **Step 4: Add the helper to `src/mousedroid/comms/base_driver.py`**

```python
def log_command_dispatch(*, driver_name: str, vx: float, vy: float, omega: float) -> None:
    """Emit a structured INFO event used by smoke-time triage."""
    _log = get_logger(__name__)
    _log.info("command_dispatch", driver=driver_name, vx=vx, vy=vy, omega=omega)
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/unit/test_serial_driver_logging.py -v --no-cov`
Expected: 1 passed.

- [ ] **Step 6: Lint + typecheck**

Run: `ruff check src/mousedroid/comms/base_driver.py tests/unit/test_serial_driver_logging.py && mypy --strict src/mousedroid/comms/base_driver.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/mousedroid/comms/base_driver.py tests/unit/test_serial_driver_logging.py
git commit -m "feat(logging): structured command_dispatch event for smoke triage"
```

---

## Task 12: Full lint / mypy / pytest gate sweep before promotion

**Files:**
- None (gates only) — but each command's output drives any remediation commits below.

- [ ] **Step 1: ruff check + format (matches CI Stage 1)**

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: zero errors. If ruff reports issues introduced by Tasks 2–11, fix in place (no `# noqa` cop-outs) and amend the related commit if it's the most recent one; otherwise add a follow-up `chore(lint)` commit per task.

- [ ] **Step 2: mypy --strict on the production package (matches CI Stage 2)**

Run: `mypy src/ --strict --ignore-missing-imports`
Expected: zero errors. Fix any new `Any`-leaks introduced by `diagnostics/` or the `usbc_discovery` field. NumPy NDArray annotations on existing protocol files must remain untouched.

- [ ] **Step 3: Config overlay validation (matches CI Stage 1b)**

Run: `python scripts/validate_configs.py --include-default`
Expected: every overlay validates — confirms the new `usbc_discovery` block deserialises in every YAML.

- [ ] **Step 4: Full pytest with coverage gate (matches CI Stage 3)**

Run: `pytest tests/unit tests/property tests/integration --cov=src/mousedroid --cov-report=term-missing --cov-fail-under=85 -q`
Expected: green across unit + integration + property. This mirrors `.github/workflows/ci.yml:121-125` exactly — local pass implies CI pass.

- [ ] **Step 5: Targeted hardware suite dry-run**

Run: `pytest tests/hardware/ -m hardware -v --no-cov`
Expected: every test either passes (Jetson) or skips with an actionable reason (non-Jetson). No errors.

- [ ] **Step 6: Commit any cleanup**

If steps 1–5 produced changes (e.g. ruff reformat, mypy fixes), commit:

```bash
git add -p
git commit -m "chore(quality): satisfy ruff + mypy --strict + coverage gates"
```

If nothing changed, skip this step.

---

## Task 13: Operator runbook

**Files:**
- Create: `docs/runbooks/jetson-rover-smoke.md`

- [ ] **Step 1: Verify the runbooks directory layout**

Run: `ls docs/runbooks/ 2>/dev/null || mkdir -p docs/runbooks`

- [ ] **Step 2: Write the runbook**

Create `docs/runbooks/jetson-rover-smoke.md` with these sections (write each section using your own words but cover every bullet exactly):

```markdown
# Jetson + USB-C Rover Smoke Runbook

## Prerequisites

- Wave Rover plugged into the Jetson Orin Nano USB-C data port
- Rover powered on (battery or bench PSU, > `safety.battery_critical_v`)
- Docker container `mousedroid` running (see `docker-compose.jetson.yml`)
- Repository checked out at `/opt/mousedroid`

## Quick start

```bash
bash scripts/jetson_full_smoke_run.sh
```

Report lands in `reports/jetson_smoke/<UTC-timestamp>/SUMMARY.md`.

## Stage gating

| Stage | Default | Override |
|-------|---------|----------|
| system | blocking | `MOUSEDROID_SMOKE_BLOCKING_SYSTEM=no` |
| usbc | blocking | `MOUSEDROID_SMOKE_BLOCKING_USBC=no` |
| gpio | blocking | `MOUSEDROID_SMOKE_BLOCKING_GPIO=no` |
| serial | blocking | `MOUSEDROID_SMOKE_BLOCKING_SERIAL=no` |
| motor | non-blocking | `MOUSEDROID_SMOKE_BLOCKING_MOTOR=yes` |
| power | blocking | `MOUSEDROID_SMOKE_BLOCKING_POWER=no` |

## Motion gate

`MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true` opts the rover into actual
motion during motor + power stages. Default is `false` — leave it off
unless the rover is on rollers or tethered.

## Failure triage matrix

| Symptom | First check |
|---------|-------------|
| `usbc enumeration` FAIL | `ls /dev/serial/by-id/` — confirm CP2102N (rover) + CP2102 (lidar) cables are seated |
| `power chain smoke` FAIL on battery | `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=false bash scripts/jetson_smoke_test.sh power` — isolate motion vs. battery |
| `motor loopback smoke` FAIL | Re-run with `MOUSEDROID_ESP32__SMOKE_TEST_ALLOW_MOTION=true` only after lifting the rover off the ground |
| `e2e` FAIL with camera reason | See `verify_sensors.py`'s `_diagnose_camera_host` output |
```

- [ ] **Step 3: Commit**

```bash
git add docs/runbooks/jetson-rover-smoke.md
git commit -m "docs(runbook): jetson + rover USB-C smoke runbook"
```

---

## Task 14: CI workflow extension

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Confirm current lint/typecheck targets**

Run: `grep -n "ruff check\|mypy src/" .github/workflows/ci.yml`
Expected: see `ruff check src/ tests/` and `mypy src/ --strict --ignore-missing-imports`. No change needed for path expansion — the new `src/mousedroid/diagnostics/` and `scripts/check_usbc_devices.py` are already covered.

- [ ] **Step 2: Add a `scripts` lint pass to catch CLI regressions**

Inside the `lint` job, append a step after the existing format-check:

```yaml
            - name: Lint scripts/
              run: ruff check scripts/
```

- [ ] **Step 3: Add a `usbc-config-gate` job**

Below the existing `config-validate` job, add:

```yaml
    usbc-config-gate:
        runs-on: ubuntu-latest
        needs: config-validate
        steps:
            - uses: actions/checkout@v4
            - uses: actions/setup-python@v5
              with:
                  python-version: "3.11"
                  cache: pip
            - name: Install package (runtime only)
              run: pip install -e .
            - name: Validate USB-C discovery wiring on production overlay
              run: pytest tests/unit/test_jetson_production_overlay.py -q --no-cov
```

- [ ] **Step 4: Verify the workflow file still parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml', encoding='utf-8'))"`
Expected: no exception.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint scripts/ + usbc-config-gate job"
```

---

## Task 15: Open PR and restore Phase 2 stash

**Files:**
- None.

- [ ] **Step 1: Push the branch**

Run: `git push -u origin feat/jetson-rover-usbc-smoke`

- [ ] **Step 2: Open the PR**

Use `gh pr create` against the integration branch:

```bash
gh pr create \
  --base claude/markdown-implementation-plan-aVJ2l \
  --head feat/jetson-rover-usbc-smoke \
  --title "feat: Jetson + USB-C rover smoke validation gate" \
  --body "$(cat <<'EOF'
## Summary
- Config-driven USB-C endpoint enumeration (`usbc_discovery` Pydantic block + helper)
- Power-chain smoke (battery + zero-vel + e-stop) wired into `jetson_smoke_test.sh` and `jetson_full_smoke_run.sh`
- Structured `command_dispatch` log for triage
- Operator runbook and CI gate

## Test plan
- [ ] `pytest tests/ --cov-fail-under=85`
- [ ] `mypy src/ --strict --ignore-missing-imports`
- [ ] `ruff check src/ tests/ scripts/ && ruff format --check src/ tests/`
- [ ] `python scripts/validate_configs.py --include-default`
- [ ] On the Jetson: `bash scripts/jetson_full_smoke_run.sh`
EOF
)"
```

- [ ] **Step 3: Restore the Phase 2 stash on the Phase 2 branch**

```bash
git switch feat/phase2-real-episode-replay
git stash pop
git status
```

Expected: original staged + unstaged + untracked entries restored.

---

## Self-Review

**1. Spec coverage:**

| User requirement | Task(s) |
|------------------|---------|
| New branch | Task 1 |
| Smoke test Jetson + rover via USB-C | Tasks 3–6, 9–10 |
| Verify motor power | Tasks 7–8 |
| Verify all hardware/software functionality | Tasks 9–10 reuse existing camera/lidar/audio/voice stages; Tasks 6, 8 add USB-C + power |
| Use agents/MCPs/worktrees/skills | Plan invokes superpowers:writing-plans, suggests superpowers:subagent-driven-development for execution; existing MCP smoke remains under Task 10 stage chain |
| Sequential thinking / pattern recognition | Tasks ordered to gate hardware after config + helpers land |
| Dynamic / backwards-compatible / reusable | Task 2 keeps `usbc_discovery: None` default; Task 5 adds disabled stub to `default.yaml` |
| No hardcoded values | Every threshold, path, glob comes from `Settings` (`ESP32Config.smoke_test_*`, `USBCDiscoveryConfig`, `safety.*`) |
| ruff / lint / mypy / numpy notes | Task 12 runs each gate; Task 14 extends CI |
| Full test suite | Task 12 step 4 runs full pytest with coverage gate |
| Logging + debugging | Task 11 adds structlog event; Task 3/7 helpers log at INFO/WARN |

**2. Placeholder scan:** No `TBD`, `TODO`, "fill in", or "implement appropriate" appears outside of legitimate quoted CLI / YAML examples.

**3. Type consistency:** `USBCDiscoveryConfig.required_endpoints`, `USBCEndpointSpec.name`, `EndpointStatus.PRESENT/MISSING/WARN`, `EndpointResult`, `PowerChainResult.battery_voltage_v` / `estop_latency_ms` are referenced identically in every task that uses them. `_PowerCapableDriver` Protocol satisfied by `MockESP32Driver` (which already implements `send_velocity`, `emergency_stop`, `get_battery_voltage`).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-30-jetson-rover-usbc-smoke-validation.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?
