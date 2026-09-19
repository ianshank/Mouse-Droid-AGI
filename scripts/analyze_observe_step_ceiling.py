#!/usr/bin/env python3
"""Amdahl consumer-ceiling bound for accelerating ``observe_step`` — desk calculation.

Task 2.1 of ``openspec/changes/mouse-droid-jetson-onnx-delivery/tasks.md``, and the
evidence for the ``performance-evidence`` requirement "The consumer ceiling SHALL be
computed before any optimization is built". **Needs no rover and no hardware**: every
number is either read from the Pydantic config or derived arithmetically from the
call-count structure of :class:`mousedroid.world_model.mcts.MCTSPlanner`.

``scripts/spike_step_distillation.py:57-61`` already implements this method for the
sibling optimization (the MCTS rollout leg) and prints the result as three *string
constants* — ``"500-650"`` calls, ``"~40%"`` rollout share, ``"~1.25-1.6x"`` ceiling.
This script computes the analogous bound for ``observe_step`` **from config**, with no
literal call counts or shares anywhere in it.

Why ``observe_step``'s ceiling is structurally worse
----------------------------------------------------
``observe_step`` runs exactly **once** per tick: ``_update_world_model`` calls it once
(``src/mousedroid/orchestrator/_world_model_state_mixin.py:41``) and ``tick()`` calls
``_update_world_model`` once (``src/mousedroid/orchestrator/orchestrator.py:521``).
``MCTSPlanner.plan()`` runs in the same tick — ``planning_hz`` is schema-only, read
nowhere in ``src/`` — and issues hundreds of ``imagine_step`` calls. So the stage being
optimized is one call competing with a planner leg two to three orders of magnitude
larger, which is a strictly smaller share and therefore a strictly worse ceiling.

``imagine_step`` calls per ``plan()`` — the derivation
------------------------------------------------------
Read off ``src/mousedroid/world_model/mcts.py``; see
:func:`imagine_calls_per_plan` for the line-by-line citations. In brief:

1. **Initial expansion** (``mcts.py:306`` -> ``mcts.py:255-257``):
   exactly ``n_action_candidates`` calls.
2. **Rollouts** (``mcts.py:325`` -> ``mcts.py:275-277``):
   exactly ``n_simulations * rollout_depth`` calls.
3. **Re-expansions** (``mcts.py:318-319``): between ``0`` and
   ``max(0, n_simulations - n_action_candidates)`` further expansions, each
   ``n_action_candidates`` calls. The exact count is a runtime property of the UCB1
   tree, not a config value, so it is reported as a bound rather than a point.

At the shipped defaults the upper bound reproduces the spike's published pair — its
``"500-650"`` calls and ``"~40%"`` rollout share — which is this derivation's anchor.

The Amdahl arithmetic
---------------------
For a stage occupying share ``s`` of end-to-end time, sped up by ``k``, the end-to-end
speedup is ``1 / ((1 - s) + s/k)``, whose limit as ``k -> inf`` is ``1 / (1 - s)``. The
ceiling depends **only on ``s``**, never on ``k``: that is the whole point.

What this script will not do
----------------------------
It does **not** invent ``s``. The measured share of tick time that ``observe_step``
occupies does not exist anywhere in the repository — ``reports/`` and
``smoke-reports/`` carry no ``tick_phase`` or ``observe_step`` figure. With no
``--observe-share`` the script sweeps a documented range and labels the whole report
conditional; see ``unknown_inputs`` in the JSON for everything the desk calculation
cannot supply.

Output
------
The JSON report is **local-only, never committed**. Its default directory
``reports/observe_step_ceiling/`` follows the repo's per-directory ignore convention
(``.gitignore:173-196``, ``:336-341``; cf. ``reports/dead_code/``), which the
``performance-evidence`` spec requires of benchmark and deploy reports. The durable
record of the number is ``proposal.md``, not this artifact.

Usage::

    # Conditional: no measured share exists, so sweep a documented range.
    python scripts/analyze_observe_step_ceiling.py --config config/jetson_production.yaml

    # Unconditional: a measured share produces a verdict and an exit code.
    python scripts/analyze_observe_step_ceiling.py --observe-share 0.08

Exit code 0 = ceiling meets ``--justify-ceiling``, or the run was conditional.
Exit code 1 = a measured share was supplied and its ceiling misses the bar.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Runnable from a bare checkout without an editable install, exactly as
# scripts/spike_step_distillation.py:48 does. Harmless under pytest, which already
# puts `src` on sys.path via [tool.pytest.ini_options].pythonpath.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousedroid.constants import MILLISECONDS_PER_SECOND
from mousedroid.logging.setup import get_logger

_log = get_logger(__name__)

# Structural call counts, read off the source rather than configured. Neither is a
# tunable: changing either means changing the tick, so they carry citations instead
# of schema fields.
OBSERVE_STEP_CALLS_PER_TICK = 1
"""``observe_step`` calls in one orchestrator tick.

``_update_world_model`` calls ``observe_step`` exactly once
(``orchestrator/_world_model_state_mixin.py:41``) and ``tick()`` calls
``_update_world_model`` exactly once (``orchestrator/orchestrator.py:521``).
"""

PLAN_CALLS_PER_TICK = 1
"""``MCTSPlanner.plan()`` calls in one orchestrator tick.

``NavigationAgent`` calls ``plan()`` once per action selection
(``agents/navigation.py:88``) and ``tick()`` selects one action per tick.
``loop.planning_hz`` is declared (``config/schema/misc.py:178``) but read nowhere in
``src/``, so planning is not throttled below ``loop.control_hz``.
"""


class CeilingInputError(ValueError):
    """An input to the ceiling calculation is outside its admissible domain."""


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlannerCallCounts:
    """``imagine_step`` calls issued by one ``MCTSPlanner.plan()`` call.

    ``total_min``/``total_max`` bracket the count because the re-expansion leg is a
    runtime property of the UCB1 tree rather than a config value.
    """

    n_simulations: int
    n_action_candidates: int
    rollout_depth: int
    initial_expansion_calls: int
    rollout_calls: int
    re_expansions_max: int
    re_expansion_calls_max: int
    total_min: int
    total_max: int

    @property
    def rollout_share_at_total_min(self) -> float:
        """Rollout leg's share of the call count when no leaf is ever re-expanded."""
        return self.rollout_calls / self.total_min

    @property
    def rollout_share_at_total_max(self) -> float:
        """Rollout leg's share of the call count when every eligible leaf expands.

        This is the quantity ``scripts/spike_step_distillation.py:60`` states as the
        string ``"~40%"``.
        """
        return self.rollout_calls / self.total_max

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping."""
        return {
            "n_simulations": self.n_simulations,
            "n_action_candidates": self.n_action_candidates,
            "rollout_depth": self.rollout_depth,
            "initial_expansion_calls": self.initial_expansion_calls,
            "rollout_calls": self.rollout_calls,
            "re_expansions_max": self.re_expansions_max,
            "re_expansion_calls_max": self.re_expansion_calls_max,
            "total_min": self.total_min,
            "total_max": self.total_max,
            "rollout_share_at_total_min": _round(self.rollout_share_at_total_min),
            "rollout_share_at_total_max": _round(self.rollout_share_at_total_max),
        }


@dataclass(frozen=True, slots=True)
class CallCountRatio:
    """``observe_step`` calls per tick against ``imagine_step`` calls per tick."""

    observe_step_calls_per_tick: int
    imagine_step_calls_per_tick_min: int
    imagine_step_calls_per_tick_max: int
    ratio_min: float
    ratio_max: float
    equal_cost_share_min: float
    equal_cost_share_max: float

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping."""
        return {
            "observe_step_calls_per_tick": self.observe_step_calls_per_tick,
            "imagine_step_calls_per_tick_min": self.imagine_step_calls_per_tick_min,
            "imagine_step_calls_per_tick_max": self.imagine_step_calls_per_tick_max,
            "ratio_min": _round(self.ratio_min),
            "ratio_max": _round(self.ratio_max),
            "equal_cost_share_min": _round(self.equal_cost_share_min),
            "equal_cost_share_max": _round(self.equal_cost_share_max),
            "equal_cost_share_meaning": (
                "Share of the tick's RSSM-transition calls that observe_step would "
                "account for IF every call cost the same. A structural reference "
                "point, NOT a measurement and NOT a bound on share of tick time: "
                "observe_step additionally runs the encoder, which pushes its share "
                "up, while the non-world-model tick phases (sense, safety, act, "
                "learn, telemetry) push it down. Net direction unknown -- which is "
                "why the default run sweeps rather than estimates."
            ),
        }


@dataclass(frozen=True, slots=True)
class FiniteSpeedup:
    """End-to-end speedup at one finite stage speedup ``k``."""

    stage_speedup: float
    end_to_end_speedup: float
    tick_saving_ms: float

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping."""
        return {
            "stage_speedup": _round(self.stage_speedup),
            "end_to_end_speedup": _round(self.end_to_end_speedup),
            "tick_saving_ms": _round(self.tick_saving_ms),
        }


@dataclass(frozen=True, slots=True)
class ShareScenario:
    """The Amdahl bound for one hypothesised stage share."""

    stage_share: float
    ceiling: float
    ceiling_tick_saving_ms: float
    finite: tuple[FiniteSpeedup, ...]

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping."""
        return {
            "stage_share": _round(self.stage_share),
            "max_end_to_end_speedup": _json_float(self.ceiling),
            "max_tick_saving_ms": _json_float(self.ceiling_tick_saving_ms),
            "at_finite_stage_speedups": [row.as_dict() for row in self.finite],
        }


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """The config values the calculation consumed, for the provenance block."""

    config_path: str
    n_simulations_base: int
    n_simulations_max: int
    rollout_depth: int
    n_action_candidates: int
    control_hz: float

    @property
    def tick_period_ms(self) -> float:
        """Tick budget in milliseconds, from ``loop.control_hz``."""
        return MILLISECONDS_PER_SECOND / self.control_hz

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping."""
        return {
            "config_path": self.config_path,
            "mcts.n_simulations_base": self.n_simulations_base,
            "mcts.n_simulations_max": self.n_simulations_max,
            "mcts.rollout_depth": self.rollout_depth,
            "mcts.n_action_candidates": self.n_action_candidates,
            "loop.control_hz": self.control_hz,
            "derived.tick_period_ms": _round(self.tick_period_ms),
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether the computed ceiling clears the justification bar."""

    conditional: bool
    justify_ceiling: float
    required_share: float
    measured_share: float | None
    achieved_ceiling: float | None
    justified: bool | None

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping."""
        return {
            "conditional": self.conditional,
            "justify_ceiling": _round(self.justify_ceiling),
            "required_stage_share": _round(self.required_share),
            "measured_stage_share": _json_float(self.measured_share),
            "achieved_ceiling": _json_float(self.achieved_ceiling),
            "justified": self.justified,
            "reading": (
                "observe_step must occupy at least required_stage_share of tick time "
                "before ANY speedup of it -- infinite included -- can reach "
                "justify_ceiling end-to-end."
            ),
        }


@dataclass(frozen=True, slots=True)
class CeilingReport:
    """The whole machine-readable report."""

    resolved_config: ResolvedConfig
    provenance: dict[str, Any]
    counts_at_base: PlannerCallCounts
    counts_at_max: PlannerCallCounts
    ratio: CallCountRatio
    scenarios: tuple[ShareScenario, ...]
    verdict: Verdict
    unknown_inputs: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-safe mapping suitable for ``json.dumps``."""
        return {
            "analysis": "observe_step consumer ceiling (Amdahl) -- desk calculation, no hardware",
            "generated_by": "scripts/analyze_observe_step_ceiling.py",
            "openspec_task": (
                "openspec/changes/mouse-droid-jetson-onnx-delivery/tasks.md 2.1 "
                "(specs/performance-evidence: consumer ceiling before optimization)"
            ),
            "tracked": False,
            "method": {
                "amdahl": "end_to_end = 1 / ((1 - s) + s/k)",
                "ceiling": "lim k->inf = 1 / (1 - s)",
                "derivation": DERIVATION,
                "sibling_precedent": (
                    "scripts/spike_step_distillation.py:57-61 states the rollout-leg "
                    "result as string constants; this script derives the observe_step "
                    "analogue from config."
                ),
            },
            "provenance": self.provenance,
            "resolved_config": self.resolved_config.as_dict(),
            "imagine_step_calls_per_plan": {
                "at_n_simulations_base": self.counts_at_base.as_dict(),
                "at_n_simulations_max": self.counts_at_max.as_dict(),
                "production_budget_note": (
                    "compute_mcts_budget (agents/_planning.py:23-24) clamps to "
                    "[base, max], and SafetyContext.surprise is never assigned from "
                    "the world model (safety/context.py:21), so the production budget "
                    "is pinned at n_simulations_base. at_n_simulations_max is the "
                    "schema ceiling, not today's behaviour."
                ),
            },
            "call_count_ratio": self.ratio.as_dict(),
            "scenarios": [row.as_dict() for row in self.scenarios],
            "verdict": self.verdict.as_dict(),
            "unknown_inputs": list(self.unknown_inputs),
            "json_float_convention": (
                "A non-finite value (an unbounded ceiling at stage_share == 1.0) "
                "serialises as null rather than the non-standard Infinity token."
            ),
        }


DERIVATION = (
    "total_min = n_action_candidates + n_simulations * rollout_depth; "
    "total_max = total_min + max(0, n_simulations - n_action_candidates) "
    "* n_action_candidates"
)
"""The call-count formula, restated for the report. See :func:`imagine_calls_per_plan`."""


# ---------------------------------------------------------------------------
# Pure arithmetic — no config, no filesystem, no torch
# ---------------------------------------------------------------------------


def _round(value: float, digits: int = 6) -> float:
    """Round for report readability, leaving non-finite values untouched."""
    if not math.isfinite(value):
        return value
    return round(value, digits)


def _json_float(value: float | None) -> float | None:
    """Map a non-finite float to ``None`` so ``json.dumps`` emits valid JSON."""
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return _round(value)


def _check_share(share: float) -> None:
    """Reject a stage share outside ``[0, 1]``.

    Raises:
        CeilingInputError: If ``share`` is NaN or outside ``[0, 1]``.
    """
    if math.isnan(share) or not 0.0 <= share <= 1.0:
        msg = f"stage share must lie in [0, 1], got {share!r}"
        raise CeilingInputError(msg)


def _check_stage_speedup(speedup: float) -> None:
    """Reject a stage speedup below ``1.0`` (``inf`` is admissible).

    Raises:
        CeilingInputError: If ``speedup`` is NaN or below 1.0.
    """
    if math.isnan(speedup) or speedup < 1.0:
        msg = f"stage speedup must be >= 1.0 (inf allowed), got {speedup!r}"
        raise CeilingInputError(msg)


def amdahl_speedup(share: float, stage_speedup: float) -> float:
    """End-to-end speedup from accelerating a stage that occupies ``share`` of the time.

    ``1 / ((1 - share) + share / stage_speedup)``.

    Args:
        share: Fraction of end-to-end time spent in the stage, in ``[0, 1]``.
        stage_speedup: Speedup of the stage alone, ``>= 1.0``; ``math.inf`` gives the
            ceiling.

    Returns:
        End-to-end speedup, always ``>= 1.0``; ``math.inf`` when the stage is the
        whole of the work and is infinitely fast.

    Raises:
        CeilingInputError: If either argument is outside its domain.
    """
    _check_share(share)
    _check_stage_speedup(stage_speedup)
    # `share / inf` is 0.0 for every finite share, so the limit falls out of the
    # same expression; only the all-serial-eliminated case needs a branch.
    accelerated = 0.0 if math.isinf(stage_speedup) else share / stage_speedup
    denominator = (1.0 - share) + accelerated
    if denominator <= 0.0:
        return math.inf
    return 1.0 / denominator


def amdahl_ceiling(share: float) -> float:
    """Limit of :func:`amdahl_speedup` as the stage speedup grows without bound.

    Args:
        share: Fraction of end-to-end time spent in the stage, in ``[0, 1]``.

    Returns:
        ``1 / (1 - share)``, or ``math.inf`` at ``share == 1``.

    Raises:
        CeilingInputError: If ``share`` is outside ``[0, 1]``.
    """
    return amdahl_speedup(share, math.inf)


def required_share_for_ceiling(target_ceiling: float) -> float:
    """Smallest stage share whose ceiling reaches ``target_ceiling``.

    Inverts :func:`amdahl_ceiling`: ``share = 1 - 1 / target_ceiling``.

    Args:
        target_ceiling: Desired end-to-end ceiling, ``>= 1.0``.

    Returns:
        The required share, in ``[0, 1)``.

    Raises:
        CeilingInputError: If ``target_ceiling`` is below 1.0 or non-finite.
    """
    if math.isnan(target_ceiling) or math.isinf(target_ceiling) or target_ceiling < 1.0:
        msg = f"target ceiling must be a finite value >= 1.0, got {target_ceiling!r}"
        raise CeilingInputError(msg)
    return 1.0 - 1.0 / target_ceiling


def imagine_calls_per_plan(
    *,
    n_simulations: int,
    n_action_candidates: int,
    rollout_depth: int,
) -> PlannerCallCounts:
    """Bound the ``imagine_step`` calls one ``MCTSPlanner.plan()`` issues.

    Every leg is read off ``src/mousedroid/world_model/mcts.py``:

    1. **Initial expansion.** ``plan()`` calls ``_expand(root, device)`` once
       (``mcts.py:306``). ``_expand`` iterates the rows of
       ``_generate_candidate_actions`` and calls ``imagine_step`` once per row
       (``mcts.py:255-257``). Both candidate strategies return exactly
       ``n_action_candidates`` rows — ``_shared_axis_candidates`` expands a
       ``linspace`` of that length (``mcts.py:145-150``) and
       ``_per_axis_candidates`` truncates its blocks with ``[:n]``
       (``mcts.py:204``) — so this leg is ``n_action_candidates`` calls and does
       **not** depend on ``action_dim``.
    2. **Rollouts.** Each of the ``n_simulations`` iterations calls
       ``_rollout(node.h, node.z, cfg.rollout_depth)`` (``mcts.py:325``), which
       calls ``imagine_step`` once per remaining depth (``mcts.py:275-277``). This
       leg is exactly ``n_simulations * rollout_depth`` calls.
    3. **Re-expansions.** A simulation expands its selected leaf only when that
       leaf has already been visited (``mcts.py:318-319``), adding another
       ``n_action_candidates`` calls. The count is bounded, not fixed:

       - *Lower bound 0.* ``_ucb1`` returns ``+inf`` for an unvisited node
         (``mcts.py:223-224``) and ``_select_child`` keeps the **first** strictly
         better score (``mcts.py:238-244``), so the first
         ``min(n_simulations, n_action_candidates)`` simulations each descend to a
         distinct root child whose ``visit_count`` is still ``0`` and skip the
         expansion branch. When ``n_simulations <= n_action_candidates`` no leaf is
         ever re-expanded.
       - *Upper bound* one expansion per remaining simulation, i.e.
         ``max(0, n_simulations - n_action_candidates)``.

       Where the true count falls inside that bracket is a runtime property of the
       UCB1 tree, not a config value, which is why it is reported as a bound.

    Args:
        n_simulations: Simulation budget for the ``plan()`` call, ``> 0``.
        n_action_candidates: ``MCTSConfig.n_action_candidates``, ``> 0``.
        rollout_depth: ``MCTSConfig.rollout_depth``, ``> 0``.

    Returns:
        The populated :class:`PlannerCallCounts`.

    Raises:
        CeilingInputError: If any argument is not a positive integer.
    """
    for name, value in (
        ("n_simulations", n_simulations),
        ("n_action_candidates", n_action_candidates),
        ("rollout_depth", rollout_depth),
    ):
        if value <= 0:
            msg = f"{name} must be > 0, got {value!r}"
            raise CeilingInputError(msg)

    initial = n_action_candidates
    rollout_calls = n_simulations * rollout_depth
    re_expansions_max = max(0, n_simulations - n_action_candidates)
    re_expansion_calls_max = re_expansions_max * n_action_candidates
    total_min = initial + rollout_calls
    return PlannerCallCounts(
        n_simulations=n_simulations,
        n_action_candidates=n_action_candidates,
        rollout_depth=rollout_depth,
        initial_expansion_calls=initial,
        rollout_calls=rollout_calls,
        re_expansions_max=re_expansions_max,
        re_expansion_calls_max=re_expansion_calls_max,
        total_min=total_min,
        total_max=total_min + re_expansion_calls_max,
    )


def call_count_ratio(counts: PlannerCallCounts) -> CallCountRatio:
    """Compare ``observe_step``'s one call per tick with the planner's call count.

    Args:
        counts: Per-``plan()`` ``imagine_step`` bounds for the tick's budget.

    Returns:
        The populated :class:`CallCountRatio`. ``equal_cost_share_*`` is the share
        ``observe_step`` would hold of the tick's RSSM-transition calls if every call
        cost the same — a structural reference point, not a measurement.
    """
    observe = OBSERVE_STEP_CALLS_PER_TICK
    imagine_min = counts.total_min * PLAN_CALLS_PER_TICK
    imagine_max = counts.total_max * PLAN_CALLS_PER_TICK
    return CallCountRatio(
        observe_step_calls_per_tick=observe,
        imagine_step_calls_per_tick_min=imagine_min,
        imagine_step_calls_per_tick_max=imagine_max,
        ratio_min=observe / imagine_max,
        ratio_max=observe / imagine_min,
        equal_cost_share_min=observe / (observe + imagine_max),
        equal_cost_share_max=observe / (observe + imagine_min),
    )


def geometric_sweep(low: float, high: float, points: int) -> tuple[float, ...]:
    """Log-spaced share samples from ``low`` to ``high`` inclusive.

    Geometric rather than linear because the plausible range spans more than two
    decades, where linear spacing would put every sample at the top end.

    Args:
        low: First sample, ``> 0``.
        high: Last sample, ``>= low``.
        points: Number of samples, ``>= 2``.

    Returns:
        Ascending tuple of ``points`` shares, every one inside ``[low, high]``.

    Raises:
        CeilingInputError: If the bounds or the sample count are inadmissible.
    """
    _check_share(low)
    _check_share(high)
    if low <= 0.0:
        msg = f"sweep low bound must be > 0 for geometric spacing, got {low!r}"
        raise CeilingInputError(msg)
    if high < low:
        msg = f"sweep high bound {high!r} is below the low bound {low!r}"
        raise CeilingInputError(msg)
    if points < 2:
        msg = f"sweep needs at least 2 points, got {points!r}"
        raise CeilingInputError(msg)
    ratio = high / low
    last = points - 1

    def sample(index: int) -> float:
        # The endpoints are pinned rather than left to `ratio ** 0` / `ratio ** 1`,
        # and every interior power is clamped into [low, high]. Both matter: when
        # `ratio` is only an ulp above 1.0, an interior power rounds a hair past
        # `high`, which would break monotonicity and could push a sample out of the
        # [0, 1] share domain that `amdahl_speedup` requires.
        if index == 0:
            return low
        if index == last:
            return high
        # `float.__pow__` is typed to return `Any` (a negative base with a
        # fractional exponent is complex), so the cast keeps the signature honest.
        raw = float(low * ratio ** (index / last))
        return min(max(raw, low), high)

    return tuple(sample(index) for index in range(points))


def build_scenario(
    share: float,
    stage_speedups: Sequence[float],
    *,
    tick_period_ms: float,
) -> ShareScenario:
    """Evaluate the Amdahl bound and the finite-``k`` speedups for one share.

    Args:
        share: Hypothesised stage share of end-to-end time, in ``[0, 1]``.
        stage_speedups: Finite stage speedups to tabulate, each ``>= 1.0``.
        tick_period_ms: Tick budget in milliseconds, ``> 0``.

    Returns:
        The populated :class:`ShareScenario`. ``tick_saving_ms`` is the wall-clock
        time removed from one tick, ``share * tick_period_ms * (1 - 1/k)``.

    Raises:
        CeilingInputError: If any argument is outside its domain.
    """
    _check_share(share)
    if math.isnan(tick_period_ms) or tick_period_ms <= 0.0:
        msg = f"tick period must be > 0 ms, got {tick_period_ms!r}"
        raise CeilingInputError(msg)
    finite = tuple(
        FiniteSpeedup(
            stage_speedup=k,
            end_to_end_speedup=amdahl_speedup(share, k),
            tick_saving_ms=share * tick_period_ms * (1.0 - 1.0 / k),
        )
        for k in stage_speedups
    )
    return ShareScenario(
        stage_share=share,
        ceiling=amdahl_ceiling(share),
        ceiling_tick_saving_ms=share * tick_period_ms,
        finite=finite,
    )


def unknown_inputs() -> tuple[str, ...]:
    """Name every input this desk calculation cannot supply without measurement.

    Returns:
        Ordered tuple of prose entries, each naming the missing quantity and where
        the measurement would have to come from.
    """
    return (
        "MEASURED SHARE OF TICK TIME OCCUPIED BY observe_step -- the one input the "
        "ceiling actually turns on. It exists nowhere in the repository: reports/ "
        "and smoke-reports/ carry no tick_phase or observe_step figure. The "
        "instrument that would produce it already exists but has never been "
        "recorded: _mark_phase('world_model', ...) "
        "(orchestrator/orchestrator.py:522) feeds "
        'mousedroid_tick_phase_ms{phase="world_model"} '
        "(telemetry/metrics/_registry_core.py:100), gated on "
        "metrics.track_tick_phases (config/schema/telemetry.py:406, default true).",
        "That 'world_model' bracket is a strict SUPERSET of observe_step: it also "
        "covers _validate_latent and the optional _latent_context blend "
        "(orchestrator/_world_model_state_mixin.py:41-50). Even once recorded it "
        "over-states observe_step's share, so it bounds the share from above rather "
        "than measuring it.",
        "Per-call cost ratio observe_step : imagine_step. observe_step additionally "
        "runs the multimodal encoder over the observation (world_model/rssm.py:107, "
        "vision gated at :131), which imagine_step never touches. The equal-cost "
        "share reported here is therefore a structural reference point, not an "
        "estimate of the real share.",
        "Where the re-expansion count falls inside [0, max(0, n_simulations - "
        "n_action_candidates)]. It is a runtime property of the UCB1 tree "
        "(mcts.py:313-322), not a config value, so the call count is a bracket.",
        "Share of tick time spent outside the world model -- sense, safety, act, "
        "learn, telemetry and the hook phases. That is the denominator of the "
        "Amdahl share and only the recorded per-phase histogram can supply it.",
        "Jetson-versus-container device ratio. Every quantity here is a call count "
        "or a pure ratio, so none of it is device-specific -- but the share it is "
        "combined with must be measured on the rover, not in a container.",
        "Whether accelerating observe_step changes the tick's critical path at all: "
        "_update_world_model is synchronous (_world_model_state_mixin.py:28) inside "
        "an async tick, so a faster call may simply return control to the event loop "
        "sooner without moving the 30 Hz deadline.",
    )


# ---------------------------------------------------------------------------
# Config + provenance (the only impure parts)
# ---------------------------------------------------------------------------


def resolve_config(config_path: Path) -> ResolvedConfig:
    """Load ``Settings`` and pull out the fields this calculation consumes.

    ``load_settings`` is imported lazily so the pure-arithmetic half of this module
    stays importable — and unit-testable — without ``pydantic_settings`` or ``torch``.

    Args:
        config_path: YAML overlay to merge over ``config/default.yaml``.

    Returns:
        The populated :class:`ResolvedConfig`.
    """
    from mousedroid.config.loader import load_settings

    overlays = (config_path,) if config_path.exists() else ()
    cfg = load_settings(*overlays)
    if not overlays:
        _log.warning("config_overlay_missing_using_defaults", path=str(config_path))
    return ResolvedConfig(
        config_path=str(config_path),
        n_simulations_base=cfg.mcts.n_simulations_base,
        n_simulations_max=cfg.mcts.n_simulations_max,
        rollout_depth=cfg.mcts.rollout_depth,
        n_action_candidates=cfg.mcts.n_action_candidates,
        control_hz=cfg.loop.control_hz,
    )


def _git(*args: str) -> str | None:
    """Run a read-only ``git`` command, returning stripped stdout or ``None``."""
    executable = shutil.which("git")
    if executable is None:
        return None
    command = [executable, *args]
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_provenance() -> dict[str, Any]:
    """Collect commit and working-tree state, host-independent by construction.

    Returns:
        Mapping with ``git_commit`` (full SHA or ``None``) and ``git_dirty``
        (``True``/``False``, or ``None`` when git is unavailable).
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "host_fields_omitted": (
            "No hostname, IP, username or device identity is recorded: the "
            "performance-evidence spec forbids committing host fingerprints, and "
            "this calculation needs none."
        ),
    }


def build_report(
    resolved: ResolvedConfig,
    *,
    shares: Sequence[float],
    stage_speedups: Sequence[float],
    justify_ceiling: float,
    measured_share: float | None,
    provenance: dict[str, Any] | None = None,
) -> CeilingReport:
    """Assemble the whole report from resolved config and CLI choices.

    Args:
        resolved: Config values the calculation consumes.
        shares: Stage shares to evaluate — one measured value, or the sweep.
        stage_speedups: Finite stage speedups to tabulate.
        justify_ceiling: End-to-end ceiling the optimization must reach.
        measured_share: The supplied measured share, or ``None`` for a conditional run.
        provenance: Pre-collected provenance; ``None`` collects it from git.

    Returns:
        The populated :class:`CeilingReport`.

    Raises:
        CeilingInputError: If any share or speedup is outside its domain.
    """
    counts_at_base = imagine_calls_per_plan(
        n_simulations=resolved.n_simulations_base,
        n_action_candidates=resolved.n_action_candidates,
        rollout_depth=resolved.rollout_depth,
    )
    counts_at_max = imagine_calls_per_plan(
        n_simulations=resolved.n_simulations_max,
        n_action_candidates=resolved.n_action_candidates,
        rollout_depth=resolved.rollout_depth,
    )
    scenarios = tuple(
        build_scenario(share, stage_speedups, tick_period_ms=resolved.tick_period_ms)
        for share in shares
    )
    verdict = Verdict(
        conditional=measured_share is None,
        justify_ceiling=justify_ceiling,
        required_share=required_share_for_ceiling(justify_ceiling),
        measured_share=measured_share,
        achieved_ceiling=None if measured_share is None else amdahl_ceiling(measured_share),
        justified=(
            None if measured_share is None else amdahl_ceiling(measured_share) >= justify_ceiling
        ),
    )
    return CeilingReport(
        resolved_config=resolved,
        provenance=git_provenance() if provenance is None else provenance,
        counts_at_base=counts_at_base,
        counts_at_max=counts_at_max,
        ratio=call_count_ratio(counts_at_base),
        scenarios=scenarios,
        verdict=verdict,
        unknown_inputs=unknown_inputs(),
    )


# ---------------------------------------------------------------------------
# CLI — a thin shell over the functions above
# ---------------------------------------------------------------------------


def parse_float_list(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated list of floats.

    Args:
        raw: Text such as ``"2,3,5,10"``.

    Returns:
        The parsed values, in order.

    Raises:
        CeilingInputError: If the list is empty or an entry is not a float.
    """
    values: list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError as exc:
            msg = f"not a number: {text!r}"
            raise CeilingInputError(msg) from exc
    if not values:
        msg = f"expected at least one number, got {raw!r}"
        raise CeilingInputError(msg)
    return tuple(values)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Amdahl consumer ceiling for accelerating observe_step. "
            "Desk calculation: no hardware, no measurement, config only."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="YAML overlay merged over config/default.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--observe-share",
        type=float,
        default=None,
        help=(
            "MEASURED share of tick time spent in observe_step, in [0, 1]. No such "
            "measurement exists in this repository, so there is deliberately no "
            "default: omitting this flag sweeps --sweep-* instead and marks the "
            "whole report conditional."
        ),
    )
    parser.add_argument(
        "--sweep-min-share",
        type=float,
        default=None,
        help=(
            "Low end of the conditional sweep. Default: derived, not hardcoded -- "
            "the equal-per-call-cost share of the tick's RSSM-transition calls, "
            "1 / (1 + imagine_step calls per tick) at the production budget."
        ),
    )
    parser.add_argument(
        "--sweep-max-share",
        type=float,
        default=0.5,
        help=(
            "High end of the conditional sweep (default: %(default)s). Half the tick "
            "is a deliberately generous hypothesis: nothing in the repository "
            "supports it, it is there so the sweep brackets the interesting region "
            "from above rather than to assert a value."
        ),
    )
    parser.add_argument(
        "--sweep-points",
        type=int,
        default=6,
        help="Log-spaced samples across the sweep range (default: %(default)s).",
    )
    parser.add_argument(
        "--stage-speedups",
        default="2,3,5,10",
        help=(
            "Comma-separated finite observe_step speedups to tabulate (default: "
            "%(default)s). 3 is the repository's existing primitive bar -- criterion "
            "1 of the GO rubric at docs/analysis/alayaworld-distillation-spike.md:65-75."
        ),
    )
    parser.add_argument(
        "--justify-ceiling",
        type=float,
        default=1.25,
        help=(
            "End-to-end ceiling the optimization must reach to be worth building "
            "(default: %(default)s). This is the LOW end of the sibling rollout "
            "leg's own ceiling, '~1.25-1.6x' at "
            "scripts/spike_step_distillation.py:61, so the bar is the one the "
            "repository already applied to the same world model."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/observe_step_ceiling/observe_step_ceiling.json"),
        help=(
            "JSON report path (default: %(default)s). The directory is gitignored "
            "per the repo's per-directory convention; the report is local-only."
        ),
    )
    return parser.parse_args(argv)


def _resolve_shares(args: argparse.Namespace, ratio: CallCountRatio) -> tuple[float, ...]:
    """Pick the measured share, or the documented sweep when none was supplied."""
    if args.observe_share is not None:
        _check_share(args.observe_share)
        return (args.observe_share,)
    low = args.sweep_min_share
    if low is None:
        low = ratio.equal_cost_share_min
    return geometric_sweep(low, args.sweep_max_share, args.sweep_points)


def _print_summary(report: CeilingReport) -> None:
    """Human-readable stdout summary — the only place this script prints.

    ``scripts/**`` is exempt from ruff ``T20`` per ``pyproject.toml:287``, and
    ``scripts/spike_step_distillation.py:330-345`` prints its ceiling the same way.
    """
    resolved = report.resolved_config
    base = report.counts_at_base
    ratio = report.ratio
    verdict = report.verdict
    print(
        f"\nCONSUMER CEILING for observe_step (config: {resolved.config_path}, "
        f"tick = {resolved.tick_period_ms:.2f} ms at {resolved.control_hz:g} Hz)"
    )
    print(
        f"  MCTS plan() imagine_step calls at n_simulations_base="
        f"{base.n_simulations}: {base.total_min}-{base.total_max} "
        f"(rollouts {base.rollout_share_at_total_max:.1%}-"
        f"{base.rollout_share_at_total_min:.1%} of them)"
    )
    print(
        f"  observe_step calls per tick: {ratio.observe_step_calls_per_tick} vs "
        f"{ratio.imagine_step_calls_per_tick_min}-"
        f"{ratio.imagine_step_calls_per_tick_max} imagine_step -- ratio "
        f"1:{1 / ratio.ratio_max:.0f} to 1:{1 / ratio.ratio_min:.0f}"
    )
    print(
        f"  equal-per-call-cost share of RSSM work: "
        f"{ratio.equal_cost_share_min:.4%}-{ratio.equal_cost_share_max:.4%} "
        f"-> ceiling {amdahl_ceiling(ratio.equal_cost_share_min):.4f}x-"
        f"{amdahl_ceiling(ratio.equal_cost_share_max):.4f}x"
    )
    print(
        "\n| observe_step share of tick | max end-to-end | at k="
        + " | at k=".join(f"{row.stage_speedup:g}" for row in report.scenarios[0].finite)
        + " |"
    )
    print("|---|---|" + "---|" * len(report.scenarios[0].finite))
    for scenario in report.scenarios:
        cells = " | ".join(f"{row.end_to_end_speedup:.3f}x" for row in scenario.finite)
        ceiling = "unbounded" if math.isinf(scenario.ceiling) else f"{scenario.ceiling:.3f}x"
        print(f"| {scenario.stage_share:.4%} | {ceiling} | {cells} |")
    print(
        f"\nTo reach {verdict.justify_ceiling:g}x end-to-end -- the low end of the "
        f"sibling rollout leg's own ceiling -- observe_step must occupy at least "
        f"{verdict.required_share:.2%} of tick time, whatever the primitive speedup."
    )
    if verdict.conditional:
        print(
            "CONDITIONAL: no --observe-share was supplied and no measured share "
            "exists in this repository, so every row above is a hypothesis, not a "
            "result. See unknown_inputs in the JSON report."
        )
    else:
        print(
            f"MEASURED share {verdict.measured_share:.4%} -> ceiling "
            f"{verdict.achieved_ceiling:.4f}x: "
            f"{'JUSTIFIED' if verdict.justified else 'NOT JUSTIFIED'}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code: 0 = conditional run, or a measured ceiling that clears
        ``--justify-ceiling``; 1 = a measured ceiling that misses it, or an
        inadmissible input.
    """
    args = _parse_args(argv)
    _log.info(
        "observe_step_ceiling_start",
        config=str(args.config),
        observe_share=args.observe_share,
        justify_ceiling=args.justify_ceiling,
        out=str(args.out),
    )
    try:
        resolved = resolve_config(args.config)
        stage_speedups = parse_float_list(args.stage_speedups)
        probe = imagine_calls_per_plan(
            n_simulations=resolved.n_simulations_base,
            n_action_candidates=resolved.n_action_candidates,
            rollout_depth=resolved.rollout_depth,
        )
        shares = _resolve_shares(args, call_count_ratio(probe))
        report = build_report(
            resolved,
            shares=shares,
            stage_speedups=stage_speedups,
            justify_ceiling=args.justify_ceiling,
            measured_share=args.observe_share,
        )
    except CeilingInputError as exc:
        _log.error("observe_step_ceiling_invalid_input", error=str(exc))
        return 1

    _log.info(
        "planner_call_counts",
        n_simulations=report.counts_at_base.n_simulations,
        total_min=report.counts_at_base.total_min,
        total_max=report.counts_at_base.total_max,
        rollout_share_at_total_max=round(report.counts_at_base.rollout_share_at_total_max, 4),
    )
    for scenario in report.scenarios:
        _log.info(
            "ceiling_scenario",
            stage_share=round(scenario.stage_share, 6),
            max_end_to_end_speedup=_json_float(scenario.ceiling),
            max_tick_saving_ms=_json_float(scenario.ceiling_tick_saving_ms),
        )
    _log.info(
        "observe_step_ceiling_verdict",
        conditional=report.verdict.conditional,
        justify_ceiling=report.verdict.justify_ceiling,
        required_stage_share=round(report.verdict.required_share, 6),
        justified=report.verdict.justified,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    _log.info("report_written", path=str(args.out))

    _print_summary(report)
    print(f"\nreport written to {args.out}")
    return 0 if report.verdict.justified is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
