"""Smoke-pass: ``validate_all_pillars`` — single-entry pillar dispatcher.

Maps each of the 10 pillars from ``docs/planning/TEN_PILLARS_VALIDATION.md``
to an async check callable and runs the full dispatch in one call. Returns
a Pydantic-typed :class:`PillarReport` that the CLI + operator runbook
both consume.

Two dispatch patterns coexist (per the approved sprint plan):

* **Pattern A — factory builder smoke**: when a ``build_*`` factory exists
  the check instantiates the subsystem and exercises a minimal smoke
  assertion. Examples: safety, world_model, memory, cognitive, reward,
  curiosity.
* **Pattern B — pytest delegation**: when no factory builder exists yet
  the check delegates to the pillar's existing unit-test module via
  in-process ``pytest.main``. The exit code maps to PASS/FAIL.
  Examples: continual (EWC + progressive), meta (MAML), scaling
  (MoE + adaptive_compute + batch_tuner), growth (distillation).

Architecture invariants (per CLAUDE.md):

* Asyncio-only at the dispatch boundary; ``pytest.main`` runs in a
  thread via ``asyncio.to_thread`` so it doesn't block the event loop.
* Structured logging via ``mousedroid.logging.setup.get_logger``.
* No hardcoded values — every threshold + test path is a module constant
  or comes from ``cfg``.
* Never raises on the happy path; per-pillar exceptions land as FAIL
  entries.
"""

from __future__ import annotations

import asyncio
import enum
import importlib.util
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mousedroid.logging.setup import get_logger

if TYPE_CHECKING:
    from mousedroid.config.schema import Settings

_log = get_logger(__name__)

# Repo root resolved from this module's location so Pattern-B test paths
# work regardless of the caller's CWD. ``__file__`` is .../src/mousedroid/
# validation/pillars.py — parents[3] is the repo root.
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]


class PillarStatus(str, enum.Enum):
    """Per-pillar + aggregate outcome (mirrors ``PreflightStatus``)."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"
    DEGRADED = "degraded"  # aggregate-only


class PillarResult(BaseModel):
    """One pillar's outcome — typed for JSON export."""

    name: str = Field(description="Pillar canonical name (safety, world_model, …).")
    status: PillarStatus
    detail: str = Field(default="", description="Human-readable diagnostic.")
    elapsed_s: float = Field(default=0.0, ge=0.0)


class PillarReport(BaseModel):
    """Aggregate pillar outcome with overall status + render helper."""

    results: list[PillarResult] = Field(default_factory=list)
    total_elapsed_s: float = Field(default=0.0, ge=0.0)

    @property
    def overall_status(self) -> PillarStatus:
        """OK when every pillar is OK (or SKIPPED); FAIL on any FAIL; DEGRADED on WARN."""
        if any(r.status == PillarStatus.FAIL for r in self.results):
            return PillarStatus.FAIL
        if any(r.status == PillarStatus.WARN for r in self.results):
            return PillarStatus.DEGRADED
        return PillarStatus.OK

    def render_text(self) -> str:
        """Render a single-screen operator summary."""
        lines = [f"Pillars: overall={self.overall_status.value}"]
        for r in self.results:
            lines.append(
                f"  [{r.status.value:8}] {r.name:14} "
                f"({r.elapsed_s * 1000.0:.1f} ms) — {r.detail}",
            )
        lines.append(
            f"Total: {self.total_elapsed_s:.2f}s across {len(self.results)} pillars",
        )
        return "\n".join(lines)


PillarCheckCallable = Callable[["Settings"], Awaitable[PillarResult]]


# Pattern B — pytest-delegation paths for pillars that don't yet have
# factory builders. Keys are pillar names; values are test-target paths.
# Centralised here (not inline) so a future test-path move only touches
# one place.
_PYTEST_DELEGATION_PATHS: dict[str, tuple[str, ...]] = {
    "continual": ("tests/unit/test_ewc.py", "tests/unit/test_progressive.py"),
    "meta": ("tests/unit/test_maml.py",),
    "scaling": (
        "tests/unit/test_moe.py",
        "tests/unit/test_adaptive_compute.py",
        "tests/unit/test_batch_tuner.py",
    ),
    "growth": ("tests/unit/test_distillation.py",),
}


def _ok(name: str, detail: str, elapsed_s: float) -> PillarResult:
    return PillarResult(
        name=name,
        status=PillarStatus.OK,
        detail=detail,
        elapsed_s=elapsed_s,
    )


def _fail(name: str, detail: str, elapsed_s: float) -> PillarResult:
    return PillarResult(
        name=name,
        status=PillarStatus.FAIL,
        detail=detail,
        elapsed_s=elapsed_s,
    )


def _skipped(name: str, detail: str) -> PillarResult:
    return PillarResult(
        name=name,
        status=PillarStatus.SKIPPED,
        detail=detail,
        elapsed_s=0.0,
    )


# ---------------------------------------------------------------------------
# Pattern A — factory-builder pillar checks
# ---------------------------------------------------------------------------


async def _check_safety(cfg: Settings) -> PillarResult:
    t0 = time.monotonic()
    from mousedroid.factory import build_safety_monitor

    monitor = build_safety_monitor(cfg)
    if monitor is None:
        return _fail("safety", "build_safety_monitor returned None", time.monotonic() - t0)
    return _ok("safety", f"monitor={type(monitor).__name__}", time.monotonic() - t0)


async def _check_world_model(cfg: Settings) -> PillarResult:
    t0 = time.monotonic()
    from mousedroid.factory import build_world_model

    wm = build_world_model(cfg)
    if wm is None:
        return _fail("world_model", "build_world_model returned None", time.monotonic() - t0)
    return _ok("world_model", f"engine={type(wm).__name__}", time.monotonic() - t0)


async def _check_memory(cfg: Settings) -> PillarResult:
    t0 = time.monotonic()
    from mousedroid.factory import build_memory_tier

    tier = build_memory_tier(cfg)
    if tier is None:
        return _skipped("memory", "build_memory_tier returned None (memory disabled in cfg)")
    return _ok("memory", f"tier={type(tier).__name__}", time.monotonic() - t0)


async def _check_cognitive(cfg: Settings) -> PillarResult:
    t0 = time.monotonic()
    from mousedroid.factory import build_cognitive_core

    core = build_cognitive_core(cfg)
    if core is None:
        return _fail("cognitive", "build_cognitive_core returned None", time.monotonic() - t0)
    return _ok("cognitive", f"core={type(core).__name__}", time.monotonic() - t0)


async def _check_reward(cfg: Settings) -> PillarResult:
    t0 = time.monotonic()
    from mousedroid.factory import build_reward_model

    model = build_reward_model(cfg)
    if model is None:
        return _fail("reward", "build_reward_model returned None", time.monotonic() - t0)
    return _ok("reward", f"model={type(model).__name__}", time.monotonic() - t0)


async def _check_curiosity(cfg: Settings) -> PillarResult:
    t0 = time.monotonic()
    from mousedroid.factory import build_curiosity_module

    module = build_curiosity_module(cfg)
    if module is None:
        return _skipped("curiosity", "build_curiosity_module returned None (disabled in cfg)")
    return _ok("curiosity", f"module={type(module).__name__}", time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Pattern B — pytest-delegation pillar checks
# ---------------------------------------------------------------------------


def _run_pytest_delegated(pillar: str, test_paths: tuple[str, ...]) -> PillarResult:
    """Run ``pytest.main`` over ``test_paths`` and map exit code to PillarResult.

    Lazy-imports ``pytest`` so the dispatch module stays importable in
    production runtimes (e.g. the Jetson Docker image) where pytest is
    a dev-only dependency. When pytest is absent, the pillar SKIPs with
    a documented reason rather than crashing the whole dispatcher.
    """
    t0 = time.monotonic()
    # Resolve each relative test path against the module-level _REPO_ROOT
    # so the dispatcher works regardless of the caller's CWD. Filter to
    # existing paths so a renamed test doesn't crash the pillar — the
    # missing path becomes part of the diagnostic message instead.
    resolved = [(_REPO_ROOT / rel) for rel in test_paths]
    existing = [str(p) for p in resolved if p.exists()]
    missing = [str(p) for p in resolved if not p.exists()]
    if not existing:
        return _fail(
            pillar,
            f"all delegated test paths missing: {missing}",
            time.monotonic() - t0,
        )

    if importlib.util.find_spec("pytest") is None:
        return _skipped(
            pillar,
            (
                "pytest not installed in this runtime (Pattern-B pillar "
                "delegation requires the dev extra; install with "
                '`pip install -e ".[dev]"` to enable).'
            ),
        )

    import pytest

    args = ["-q", "--no-header", "-x", *existing]
    exit_code = pytest.main(args)
    elapsed = time.monotonic() - t0
    if exit_code == pytest.ExitCode.OK:
        detail = f"delegated to {len(existing)} test module(s)"
        if missing:
            detail += f" (missing: {missing})"
        return _ok(pillar, detail, elapsed)
    return _fail(
        pillar,
        f"pytest exit_code={int(exit_code)} on {existing}",
        elapsed,
    )


async def _check_continual(cfg: Settings) -> PillarResult:
    del cfg  # unused — pytest delegation reads its own cfg
    return await asyncio.to_thread(
        _run_pytest_delegated,
        "continual",
        _PYTEST_DELEGATION_PATHS["continual"],
    )


async def _check_meta(cfg: Settings) -> PillarResult:
    del cfg
    return await asyncio.to_thread(
        _run_pytest_delegated,
        "meta",
        _PYTEST_DELEGATION_PATHS["meta"],
    )


async def _check_scaling(cfg: Settings) -> PillarResult:
    del cfg
    return await asyncio.to_thread(
        _run_pytest_delegated,
        "scaling",
        _PYTEST_DELEGATION_PATHS["scaling"],
    )


async def _check_growth(cfg: Settings) -> PillarResult:
    del cfg
    return await asyncio.to_thread(
        _run_pytest_delegated,
        "growth",
        _PYTEST_DELEGATION_PATHS["growth"],
    )


# Canonical dispatch table — pillar name → check callable.
# Order matches ``docs/planning/TEN_PILLARS_VALIDATION.md``.
_PILLAR_DISPATCH: dict[str, PillarCheckCallable] = {
    "safety": _check_safety,
    "world_model": _check_world_model,
    "memory": _check_memory,
    "cognitive": _check_cognitive,
    "reward": _check_reward,
    "curiosity": _check_curiosity,
    "continual": _check_continual,
    "meta": _check_meta,
    "scaling": _check_scaling,
    "growth": _check_growth,
}


async def validate_all_pillars(
    cfg: Settings,
    *,
    pillar_names: set[str] | None = None,
    dry_run: bool = False,
) -> PillarReport:
    """Run all (or filtered) pillar checks and return the aggregate report.

    Args:
        cfg: Loaded :class:`Settings`.
        pillar_names: When set, only run pillars whose names are in this set.
        dry_run: When True, list pillars without invoking checks — every
            entry returns ``SKIPPED`` with ``detail="dry-run"`` so CI can
            sanity-check the dispatch table without paying the per-pillar
            wall-clock.

    Returns:
        :class:`PillarReport` — typed, exportable to JSON or text.
    """
    names = pillar_names if pillar_names is not None else set(_PILLAR_DISPATCH.keys())
    selected = [(name, fn) for name, fn in _PILLAR_DISPATCH.items() if name in names]
    _log.info("pillar_validation_start", pillars=[n for n, _ in selected], dry_run=dry_run)

    t0 = time.monotonic()
    results: list[PillarResult] = []
    for name, fn in selected:
        if dry_run:
            results.append(_skipped(name, "dry-run"))
            continue
        try:
            result = await fn(cfg)
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning(
                "pillar_check_exception",
                pillar=name,
                error=f"{type(exc).__name__}:{exc}",
            )
            result = PillarResult(
                name=name,
                status=PillarStatus.FAIL,
                detail=f"{type(exc).__name__}: {exc}",
                elapsed_s=0.0,
            )
        results.append(result)

    report = PillarReport(results=results, total_elapsed_s=time.monotonic() - t0)
    _log.info(
        "pillar_validation_complete",
        overall=report.overall_status.value,
        elapsed_s=report.total_elapsed_s,
        pillars_run=len(results),
        dry_run=dry_run,
    )
    return report


__all__ = [
    "PillarReport",
    "PillarResult",
    "PillarStatus",
    "validate_all_pillars",
]
