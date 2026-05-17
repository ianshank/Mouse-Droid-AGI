"""Smoke-pass: programmatic pre-flight validation.

Wraps the existing :mod:`mousedroid.validation.runtime` helpers (used today
only by ``scripts/verify_sensors.py`` and ``scripts/preflight_check.sh``) in
a single async ``run_preflight(cfg) -> PreflightReport`` entry point that
the orchestrator and the ``validate_all_pillars`` CLI both consume. Replaces
the shell-only path with a Pydantic-typed report so operator dashboards
and PR templates can parse the outcome programmatically.

Architecture invariants (per CLAUDE.md):

* Asyncio-only — every check helper is ``async``. No blocking I/O.
* Structured logging via ``mousedroid.logging.setup.get_logger``.
* No hardcoded thresholds — every check resolves its expectations from
  the injected :class:`Settings` (e.g. lidar coverage threshold lives at
  ``cfg.lidar.min_scan_coverage_deg``).
* Never raises on the happy path; per-check exceptions are caught at the
  dispatch layer and recorded as FAIL entries so a misbehaving driver
  never crashes the operator runbook.
* Reuses existing :mod:`mousedroid.validation.runtime` helpers and
  :mod:`mousedroid.factory` builders — no parallel sensor code paths.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings

_log = get_logger(__name__)


class PreflightStatus(str, enum.Enum):
    """Outcome shared by individual checks + the aggregate report.

    Per-check checks may return ``OK | WARN | FAIL``. The aggregate
    :class:`PreflightReport.overall_status` additionally promotes any
    WARN-but-no-FAIL aggregate to ``DEGRADED`` so dashboards can route
    on "everything works but watch this" vs "all green".
    """

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    DEGRADED = "degraded"  # aggregate-only


class PreflightCheckResult(BaseModel):
    """One preflight check outcome — typed for JSON export to operator tooling."""

    name: str = Field(description="Subsystem name (camera, microphone, lidar, …).")
    status: PreflightStatus
    detail: str = Field(default="", description="Human-readable diagnostic.")
    elapsed_s: float = Field(default=0.0, ge=0.0)


class PreflightReport(BaseModel):
    """Aggregate preflight outcome with overall status + render helper."""

    checks: list[PreflightCheckResult] = Field(default_factory=list)
    total_elapsed_s: float = Field(default=0.0, ge=0.0)

    @property
    def overall_status(self) -> PreflightStatus:
        """OK only when every check is OK; FAIL on any FAIL; DEGRADED on WARN."""
        if any(c.status == PreflightStatus.FAIL for c in self.checks):
            return PreflightStatus.FAIL
        if any(c.status == PreflightStatus.WARN for c in self.checks):
            return PreflightStatus.DEGRADED
        return PreflightStatus.OK

    def render_text(self) -> str:
        """Render a single-screen operator summary (newline-separated)."""
        lines = [f"Preflight: overall={self.overall_status.value}"]
        for c in self.checks:
            lines.append(
                f"  [{c.status.value:5}] {c.name:12} "
                f"({c.elapsed_s * 1000.0:.1f} ms) — {c.detail}",
            )
        lines.append(
            f"Total: {self.total_elapsed_s:.2f}s across {len(self.checks)} checks",
        )
        return "\n".join(lines)


CheckCallable = Callable[["Settings"], Awaitable[PreflightCheckResult]]


def _ok(name: str, detail: str, elapsed_s: float) -> PreflightCheckResult:
    return PreflightCheckResult(
        name=name, status=PreflightStatus.OK, detail=detail, elapsed_s=elapsed_s,
    )


def _fail(name: str, detail: str, elapsed_s: float) -> PreflightCheckResult:
    return PreflightCheckResult(
        name=name, status=PreflightStatus.FAIL, detail=detail, elapsed_s=elapsed_s,
    )


def _warn(name: str, detail: str, elapsed_s: float) -> PreflightCheckResult:
    return PreflightCheckResult(
        name=name, status=PreflightStatus.WARN, detail=detail, elapsed_s=elapsed_s,
    )


async def _check_camera(cfg: Settings) -> PreflightCheckResult:
    """Probe the camera (mock_hardware short-circuits to OK)."""
    t0 = time.monotonic()
    if cfg.mock_hardware:
        return _ok("camera", "mock_hardware=true", time.monotonic() - t0)

    from mousedroid.validation.runtime import (
        camera_unavailable_reason,
        capture_camera_frame,
    )

    reason = camera_unavailable_reason(cfg)
    if reason is not None:
        return _fail("camera", reason, time.monotonic() - t0)
    frame, source = await capture_camera_frame(cfg)
    return _ok(
        "camera",
        f"source={source} shape={getattr(frame, 'shape', None)}",
        time.monotonic() - t0,
    )


async def _check_microphone(cfg: Settings) -> PreflightCheckResult:
    """Probe the microphone (mock_hardware short-circuits to OK)."""
    t0 = time.monotonic()
    if cfg.mock_hardware:
        return _ok("microphone", "mock_hardware=true", time.monotonic() - t0)

    from mousedroid.validation.runtime import capture_microphone_chunk

    chunk = await capture_microphone_chunk(cfg)
    if chunk is None:
        return _warn(
            "microphone",
            "no audio captured (driver returned None)",
            time.monotonic() - t0,
        )
    return _ok(
        "microphone", f"samples={chunk.size} dtype={chunk.dtype}", time.monotonic() - t0,
    )


async def _check_speaker(cfg: Settings) -> PreflightCheckResult:
    """Probe the speaker (mock_hardware short-circuits to OK)."""
    t0 = time.monotonic()
    if cfg.mock_hardware:
        return _ok("speaker", "mock_hardware=true", time.monotonic() - t0)

    from mousedroid.validation.runtime import play_speaker_tone

    samples_written = await play_speaker_tone(cfg)
    return _ok("speaker", f"samples_written={samples_written}", time.monotonic() - t0)


async def _check_lidar(cfg: Settings) -> PreflightCheckResult:
    """Probe the LiDAR (mock_hardware short-circuits to OK).

    ``collect_lidar_diagnostics`` returns a ``list[LidarScanDiagnostics]``
    (one entry per scan). We aggregate across the list — total points
    must be non-zero and the worst-case validation_coverage_deg must
    meet ``cfg.lidar.min_scan_coverage_deg``.
    """
    t0 = time.monotonic()
    if cfg.mock_hardware:
        return _ok("lidar", "mock_hardware=true", time.monotonic() - t0)

    from mousedroid.validation.runtime import collect_lidar_diagnostics

    diagnostics = await collect_lidar_diagnostics(cfg)
    if not diagnostics:
        return _fail("lidar", "no diagnostics returned", time.monotonic() - t0)

    total_points = sum(d.n_points for d in diagnostics)
    if total_points == 0:
        return _fail("lidar", "no scan points captured", time.monotonic() - t0)

    worst_coverage = min(d.validation_coverage_deg for d in diagnostics)
    min_coverage = getattr(cfg.lidar, "min_scan_coverage_deg", 0.0)
    if worst_coverage < min_coverage:
        return _warn(
            "lidar",
            (
                f"worst validation_coverage_deg={worst_coverage:.1f} below "
                f"min_scan_coverage_deg={min_coverage:.1f}"
            ),
            time.monotonic() - t0,
        )
    return _ok(
        "lidar",
        f"scans={len(diagnostics)} total_points={total_points} "
        f"worst_coverage_deg={worst_coverage:.1f}",
        time.monotonic() - t0,
    )


async def _check_esp32(cfg: Settings) -> PreflightCheckResult:
    """Probe the ESP32 motor driver (mock_hardware short-circuits to OK)."""
    t0 = time.monotonic()
    if cfg.mock_hardware:
        return _ok("esp32", "mock_hardware=true", time.monotonic() - t0)

    from mousedroid.factory import build_esp32_driver

    driver = build_esp32_driver(cfg)
    if driver is None:
        return _fail("esp32", "build_esp32_driver returned None", time.monotonic() - t0)
    return _ok("esp32", f"driver={type(driver).__name__}", time.monotonic() - t0)


async def _check_config(cfg: Settings) -> PreflightCheckResult:
    """Sanity-check core config invariants (always runs, mock or not)."""
    t0 = time.monotonic()
    issues: list[str] = []
    if cfg.model.action_dim <= 0:
        issues.append("model.action_dim must be > 0")
    if cfg.loop.control_hz <= 0:
        issues.append("loop.control_hz must be > 0")
    if issues:
        return _fail("config", "; ".join(issues), time.monotonic() - t0)
    return _ok(
        "config",
        f"action_dim={cfg.model.action_dim} control_hz={cfg.loop.control_hz}",
        time.monotonic() - t0,
    )


_CHECK_DISPATCH: dict[str, CheckCallable] = {
    "camera": _check_camera,
    "microphone": _check_microphone,
    "speaker": _check_speaker,
    "lidar": _check_lidar,
    "esp32": _check_esp32,
    "config": _check_config,
}


async def run_preflight(
    cfg: Settings,
    *,
    check_names: set[str] | None = None,
) -> PreflightReport:
    """Run all (or filtered) preflight checks and return the aggregate report.

    Per-check exceptions are caught and recorded as ``FAIL`` entries — the
    operator runbook never bubbles a driver crash.

    Args:
        cfg: Loaded :class:`Settings`.
        check_names: When set, only run checks whose names are in this set.

    Returns:
        :class:`PreflightReport` — typed, exportable to JSON or text.
    """
    names = check_names if check_names is not None else set(_CHECK_DISPATCH.keys())
    selected = [(name, fn) for name, fn in _CHECK_DISPATCH.items() if name in names]
    _log.info("preflight_start", checks=[n for n, _ in selected])

    t0 = time.monotonic()
    results: list[PreflightCheckResult] = []
    for name, fn in selected:
        try:
            result = await fn(cfg)
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning(
                "preflight_check_exception",
                check=name,
                error=f"{type(exc).__name__}:{exc}",
            )
            result = PreflightCheckResult(
                name=name,
                status=PreflightStatus.FAIL,
                detail=f"{type(exc).__name__}: {exc}",
                elapsed_s=0.0,
            )
        results.append(result)

    report = PreflightReport(checks=results, total_elapsed_s=time.monotonic() - t0)
    _log.info(
        "preflight_complete",
        overall=report.overall_status.value,
        elapsed_s=report.total_elapsed_s,
        checks_run=len(results),
    )
    return report


__all__ = [
    "PreflightCheckResult",
    "PreflightReport",
    "PreflightStatus",
    "run_preflight",
]
