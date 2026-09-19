"""Unit tests for the observe_step consumer-ceiling desk calculation.

Covers ``scripts/analyze_observe_step_ceiling.py`` (openspec task 2.1 of
``mouse-droid-jetson-onnx-delivery``): the Amdahl arithmetic including the
``k -> inf`` limit and the ``s = 0`` / ``s = 1`` edges, the ``imagine_step``
call-count derivation against the shipped ``MCTSConfig`` values, argument
validation, and the JSON report's schema.

Deliberately dependency-light: the script under test imports only
``mousedroid.constants`` and ``mousedroid.logging.setup`` at module scope and
defers ``load_settings`` into ``resolve_config``, so these tests need neither
``torch`` nor ``onnxruntime``. The pure-arithmetic tests touch no filesystem at
all; the two CLI tests write only under ``tmp_path``.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

from tests._script_loader import load_script_module

_mod = load_script_module("analyze_observe_step_ceiling")

# The shipped MCTSConfig defaults (src/mousedroid/config/schema/world_model.py:273-277).
# Restated here as the expected-value fixture the derivation is checked against --
# the point of these tests is that the script computes them rather than storing them.
_N_SIMULATIONS_BASE = 50
_N_SIMULATIONS_MAX = 200
_ROLLOUT_DEPTH = 5
_N_ACTION_CANDIDATES = 9
_CONTROL_HZ = 30.0


def _counts(
    *,
    n_simulations: int = _N_SIMULATIONS_BASE,
    n_action_candidates: int = _N_ACTION_CANDIDATES,
    rollout_depth: int = _ROLLOUT_DEPTH,
) -> object:
    return _mod.imagine_calls_per_plan(
        n_simulations=n_simulations,
        n_action_candidates=n_action_candidates,
        rollout_depth=rollout_depth,
    )


def _resolved(config_path: str = "config/default.yaml") -> object:
    return _mod.ResolvedConfig(
        config_path=config_path,
        n_simulations_base=_N_SIMULATIONS_BASE,
        n_simulations_max=_N_SIMULATIONS_MAX,
        rollout_depth=_ROLLOUT_DEPTH,
        n_action_candidates=_N_ACTION_CANDIDATES,
        control_hz=_CONTROL_HZ,
    )


_FAKE_PROVENANCE = {"git_commit": "0" * 40, "git_dirty": False}


class TestAmdahlSpeedup:
    """``1 / ((1 - s) + s/k)``."""

    @pytest.mark.parametrize(
        ("share", "speedup", "expected"),
        [
            (0.5, 2.0, 4.0 / 3.0),
            (0.5, 10.0, 1.0 / 0.55),
            (0.2, 4.0, 1.0 / 0.85),
            (0.9, 3.0, 1.0 / 0.4),
        ],
    )
    def test_matches_closed_form(self, share: float, speedup: float, expected: float) -> None:
        assert _mod.amdahl_speedup(share, speedup) == pytest.approx(expected)

    def test_k_of_one_is_a_no_op(self) -> None:
        for share in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert _mod.amdahl_speedup(share, 1.0) == pytest.approx(1.0)

    def test_infinite_k_is_the_ceiling(self) -> None:
        for share in (0.0, 0.001, 0.2, 0.5, 0.99):
            assert _mod.amdahl_speedup(share, math.inf) == pytest.approx(1.0 / (1.0 - share))

    def test_large_finite_k_converges_to_the_ceiling(self) -> None:
        share = 0.4
        ceiling = _mod.amdahl_ceiling(share)
        assert _mod.amdahl_speedup(share, 1e9) == pytest.approx(ceiling, rel=1e-6)
        assert _mod.amdahl_speedup(share, 1e9) < ceiling

    def test_zero_share_never_helps(self) -> None:
        for speedup in (1.0, 2.0, 1e6, math.inf):
            assert _mod.amdahl_speedup(0.0, speedup) == pytest.approx(1.0)
        assert _mod.amdahl_ceiling(0.0) == pytest.approx(1.0)

    def test_unit_share_returns_the_stage_speedup(self) -> None:
        for speedup in (1.0, 2.0, 7.5, 1e6):
            assert _mod.amdahl_speedup(1.0, speedup) == pytest.approx(speedup)

    def test_unit_share_ceiling_is_unbounded(self) -> None:
        assert _mod.amdahl_speedup(1.0, math.inf) == math.inf
        assert _mod.amdahl_ceiling(1.0) == math.inf


class TestRequiredShare:
    """``amdahl_ceiling`` inverted."""

    def test_inverts_the_ceiling(self) -> None:
        for target in (1.0, 1.25, 1.6, 2.0, 10.0):
            share = _mod.required_share_for_ceiling(target)
            assert _mod.amdahl_ceiling(share) == pytest.approx(target)

    def test_repo_justification_bar_needs_a_fifth_of_the_tick(self) -> None:
        # 1.25x is the low end of the sibling rollout leg's ceiling
        # (scripts/spike_step_distillation.py:61).
        assert _mod.required_share_for_ceiling(1.25) == pytest.approx(0.2)

    @pytest.mark.parametrize("target", [0.0, 0.99, -1.0, math.inf, math.nan])
    def test_rejects_inadmissible_targets(self, target: float) -> None:
        with pytest.raises(_mod.CeilingInputError):
            _mod.required_share_for_ceiling(target)


class TestCallCountDerivation:
    """``imagine_step`` calls per ``plan()``, from MCTSConfig."""

    def test_legs_at_shipped_defaults(self) -> None:
        counts = _counts()
        # Initial expansion: mcts.py:306 -> mcts.py:255-257, one call per candidate.
        assert counts.initial_expansion_calls == _N_ACTION_CANDIDATES
        # Rollouts: mcts.py:325 -> mcts.py:275-277, depth calls per simulation.
        assert counts.rollout_calls == _N_SIMULATIONS_BASE * _ROLLOUT_DEPTH == 250
        # Re-expansions: mcts.py:318-319, at most one per simulation beyond the
        # first n_action_candidates (each of which lands on an unvisited root child).
        assert counts.re_expansions_max == _N_SIMULATIONS_BASE - _N_ACTION_CANDIDATES == 41
        assert counts.re_expansion_calls_max == 41 * _N_ACTION_CANDIDATES == 369

    def test_totals_at_shipped_defaults(self) -> None:
        counts = _counts()
        assert counts.total_min == 259  # 9 + 250
        assert counts.total_max == 628  # 9 + 250 + 369

    def test_upper_bound_reproduces_the_spike_anchor(self) -> None:
        """The published pair at scripts/spike_step_distillation.py:59-60.

        The spike states ``"500-650"`` calls and a ``"~40%"`` rollout share as
        string constants. The derivation must land inside that range and on that
        share, or it is not the same method applied to a different stage.
        """
        counts = _counts()
        assert 500 <= counts.total_max <= 650
        assert counts.rollout_share_at_total_max == pytest.approx(0.40, abs=0.01)

    def test_totals_at_the_schema_maximum_budget(self) -> None:
        counts = _counts(n_simulations=_N_SIMULATIONS_MAX)
        assert counts.rollout_calls == 1000
        assert counts.total_min == 1009
        assert counts.total_max == 2728  # 1009 + 191 * 9

    def test_no_re_expansion_when_budget_fits_the_candidate_set(self) -> None:
        """mcts.py:223-224 + :238-244 — the first C sims each hit an unvisited child."""
        for budget in (1, _N_ACTION_CANDIDATES - 1, _N_ACTION_CANDIDATES):
            counts = _counts(n_simulations=budget)
            assert counts.re_expansions_max == 0
            assert counts.re_expansion_calls_max == 0
            assert counts.total_min == counts.total_max

    def test_derivation_takes_no_action_dim(self) -> None:
        """Both candidate strategies return exactly n_action_candidates rows.

        ``_shared_axis_candidates`` expands a ``linspace`` of that length
        (mcts.py:145-150) and ``_per_axis_candidates`` truncates with ``[:n]``
        (mcts.py:204), so the expansion leg costs ``n_action_candidates`` calls
        whatever ``action_dim`` is — the derivation must not accept one at all.
        """
        params = set(inspect.signature(_mod.imagine_calls_per_plan).parameters)
        assert params == {"n_simulations", "n_action_candidates", "rollout_depth"}

    @pytest.mark.parametrize(
        ("n_simulations", "n_action_candidates", "rollout_depth"),
        [(0, 9, 5), (50, 0, 5), (50, 9, 0), (-1, 9, 5)],
    )
    def test_rejects_non_positive_counts(
        self, n_simulations: int, n_action_candidates: int, rollout_depth: int
    ) -> None:
        with pytest.raises(_mod.CeilingInputError):
            _counts(
                n_simulations=n_simulations,
                n_action_candidates=n_action_candidates,
                rollout_depth=rollout_depth,
            )


class TestCallCountRatio:
    """One ``observe_step`` per tick against the planner's call count."""

    def test_ratio_at_shipped_defaults(self) -> None:
        ratio = _mod.call_count_ratio(_counts())
        assert ratio.observe_step_calls_per_tick == 1
        assert ratio.imagine_step_calls_per_tick_min == 259
        assert ratio.imagine_step_calls_per_tick_max == 628
        assert ratio.ratio_min == pytest.approx(1 / 628)
        assert ratio.ratio_max == pytest.approx(1 / 259)

    def test_equal_cost_share_is_three_orders_below_the_bar(self) -> None:
        ratio = _mod.call_count_ratio(_counts())
        assert ratio.equal_cost_share_min == pytest.approx(1 / 629)
        assert ratio.equal_cost_share_max == pytest.approx(1 / 260)
        assert _mod.amdahl_ceiling(ratio.equal_cost_share_max) < 1.01


class TestShareValidation:
    """Out-of-range shares and speedups are rejected, not clamped."""

    @pytest.mark.parametrize("share", [-0.001, 1.001, -1.0, 2.0, math.nan, math.inf])
    def test_rejects_out_of_range_share(self, share: float) -> None:
        with pytest.raises(_mod.CeilingInputError, match="stage share"):
            _mod.amdahl_speedup(share, 2.0)

    @pytest.mark.parametrize("speedup", [0.0, 0.5, -2.0, math.nan])
    def test_rejects_stage_speedup_below_one(self, speedup: float) -> None:
        with pytest.raises(_mod.CeilingInputError, match="stage speedup"):
            _mod.amdahl_speedup(0.5, speedup)

    def test_build_scenario_rejects_non_positive_tick_period(self) -> None:
        with pytest.raises(_mod.CeilingInputError, match="tick period"):
            _mod.build_scenario(0.1, (2.0,), tick_period_ms=0.0)


class TestGeometricSweep:
    def test_spans_the_range_inclusively(self) -> None:
        sweep = _mod.geometric_sweep(0.001, 0.5, 4)
        assert len(sweep) == 4
        assert sweep[0] == pytest.approx(0.001)
        assert sweep[-1] == pytest.approx(0.5)

    def test_is_ascending_and_log_spaced(self) -> None:
        sweep = _mod.geometric_sweep(0.002, 0.5, 5)
        assert list(sweep) == sorted(sweep)
        ratios = [sweep[i + 1] / sweep[i] for i in range(len(sweep) - 1)]
        for ratio in ratios[1:]:
            assert ratio == pytest.approx(ratios[0])

    def test_near_degenerate_range_stays_monotone_and_in_bounds(self) -> None:
        """Regression pin for a bug the property tier found.

        With ``high / low`` only one ulp above 1.0, the interior power
        ``low * ratio ** (i / last)`` rounds a hair *above* the pinned ``high``,
        which broke monotonicity and could hand ``amdahl_speedup`` a share outside
        ``[0, 1]``. Interior samples are now clamped into ``[low, high]``.
        """
        low, high = 0.39999999999999997, 0.4
        sweep = _mod.geometric_sweep(low, high, 4)
        assert list(sweep) == sorted(sweep)
        for share in sweep:
            assert low <= share <= high

    def test_equal_bounds_collapse_to_a_constant_sweep(self) -> None:
        assert _mod.geometric_sweep(0.25, 0.25, 3) == (0.25, 0.25, 0.25)

    @pytest.mark.parametrize(
        ("low", "high", "points"),
        [(0.0, 0.5, 4), (0.5, 0.1, 4), (0.01, 0.5, 1), (-0.1, 0.5, 4), (0.01, 1.5, 4)],
    )
    def test_rejects_inadmissible_bounds(self, low: float, high: float, points: int) -> None:
        with pytest.raises(_mod.CeilingInputError):
            _mod.geometric_sweep(low, high, points)


class TestParseFloatList:
    def test_parses_and_preserves_order(self) -> None:
        assert _mod.parse_float_list("2,3,5,10") == (2.0, 3.0, 5.0, 10.0)

    def test_tolerates_whitespace_and_trailing_commas(self) -> None:
        assert _mod.parse_float_list(" 2 , 3.5 , ") == (2.0, 3.5)

    @pytest.mark.parametrize("raw", ["", ",", "2,abc", "n/a"])
    def test_rejects_unparseable_input(self, raw: str) -> None:
        with pytest.raises(_mod.CeilingInputError):
            _mod.parse_float_list(raw)


class TestScenario:
    def test_finite_rows_and_savings(self) -> None:
        scenario = _mod.build_scenario(0.5, (2.0, 4.0), tick_period_ms=100.0)
        assert scenario.stage_share == 0.5
        assert scenario.ceiling == pytest.approx(2.0)
        # An infinitely fast stage removes its whole slice of the tick and no more.
        assert scenario.ceiling_tick_saving_ms == pytest.approx(50.0)
        assert [row.stage_speedup for row in scenario.finite] == [2.0, 4.0]
        assert scenario.finite[0].end_to_end_speedup == pytest.approx(4.0 / 3.0)
        assert scenario.finite[0].tick_saving_ms == pytest.approx(25.0)
        assert scenario.finite[1].tick_saving_ms == pytest.approx(37.5)

    def test_saving_never_exceeds_the_stage_slice(self) -> None:
        scenario = _mod.build_scenario(0.25, (2.0, 10.0, 1000.0), tick_period_ms=33.0)
        for row in scenario.finite:
            assert row.tick_saving_ms < scenario.ceiling_tick_saving_ms


class TestUnknownInputs:
    def test_names_the_missing_measurement_first(self) -> None:
        entries = _mod.unknown_inputs()
        assert len(entries) >= 3
        assert "MEASURED SHARE OF TICK TIME" in entries[0]
        assert "tick_phase" in entries[0]

    def test_every_entry_is_non_empty_prose(self) -> None:
        for entry in _mod.unknown_inputs():
            assert isinstance(entry, str)
            assert len(entry) > 40


class TestReportSchema:
    """The machine-readable report's keys and JSON-safety."""

    def _report(self, *, measured_share: float | None) -> object:
        return _mod.build_report(
            _resolved(),
            shares=(0.1, 0.2) if measured_share is None else (measured_share,),
            stage_speedups=(2.0, 3.0),
            justify_ceiling=1.25,
            measured_share=measured_share,
            provenance=_FAKE_PROVENANCE,
        )

    def test_top_level_keys(self) -> None:
        payload = self._report(measured_share=None).as_dict()
        assert set(payload) == {
            "analysis",
            "generated_by",
            "openspec_task",
            "tracked",
            "method",
            "provenance",
            "resolved_config",
            "imagine_step_calls_per_plan",
            "call_count_ratio",
            "scenarios",
            "verdict",
            "unknown_inputs",
            "json_float_convention",
        }

    def test_report_is_json_serialisable_and_round_trips(self) -> None:
        payload = self._report(measured_share=None).as_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_declares_itself_untracked(self) -> None:
        assert self._report(measured_share=None).as_dict()["tracked"] is False

    def test_provenance_is_passed_through_without_host_fields(self) -> None:
        payload = self._report(measured_share=None).as_dict()
        assert payload["provenance"] == _FAKE_PROVENANCE

    def test_resolved_config_records_every_consumed_field(self) -> None:
        resolved = self._report(measured_share=None).as_dict()["resolved_config"]
        assert resolved["mcts.n_simulations_base"] == _N_SIMULATIONS_BASE
        assert resolved["mcts.n_simulations_max"] == _N_SIMULATIONS_MAX
        assert resolved["mcts.rollout_depth"] == _ROLLOUT_DEPTH
        assert resolved["mcts.n_action_candidates"] == _N_ACTION_CANDIDATES
        assert resolved["loop.control_hz"] == _CONTROL_HZ
        assert resolved["derived.tick_period_ms"] == pytest.approx(1000.0 / _CONTROL_HZ)

    def test_both_budgets_are_reported(self) -> None:
        calls = self._report(measured_share=None).as_dict()["imagine_step_calls_per_plan"]
        assert calls["at_n_simulations_base"]["total_max"] == 628
        assert calls["at_n_simulations_max"]["total_max"] == 2728

    def test_scenario_keys(self) -> None:
        scenario = self._report(measured_share=None).as_dict()["scenarios"][0]
        assert set(scenario) == {
            "stage_share",
            "max_end_to_end_speedup",
            "max_tick_saving_ms",
            "at_finite_stage_speedups",
        }
        assert set(scenario["at_finite_stage_speedups"][0]) == {
            "stage_speedup",
            "end_to_end_speedup",
            "tick_saving_ms",
        }

    def test_conditional_run_invents_no_share(self) -> None:
        verdict = self._report(measured_share=None).as_dict()["verdict"]
        assert verdict["conditional"] is True
        assert verdict["measured_stage_share"] is None
        assert verdict["achieved_ceiling"] is None
        assert verdict["justified"] is None
        assert verdict["required_stage_share"] == pytest.approx(0.2)

    def test_measured_share_below_the_bar_is_not_justified(self) -> None:
        verdict = self._report(measured_share=0.08).as_dict()["verdict"]
        assert verdict["conditional"] is False
        assert verdict["achieved_ceiling"] == pytest.approx(1.0 / 0.92, rel=1e-4)
        assert verdict["justified"] is False

    def test_measured_share_above_the_bar_is_justified(self) -> None:
        verdict = self._report(measured_share=0.3).as_dict()["verdict"]
        assert verdict["justified"] is True

    def test_unbounded_ceiling_serialises_as_null(self) -> None:
        payload = self._report(measured_share=1.0).as_dict()
        assert payload["scenarios"][0]["max_end_to_end_speedup"] is None
        assert payload["verdict"]["achieved_ceiling"] is None
        json.dumps(payload)  # would raise on a bare Infinity token


class TestCli:
    """The CLI is a thin shell: it resolves config, delegates, writes, exits."""

    def test_conditional_run_writes_a_report_and_exits_zero(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "ceiling.json"
        assert _mod.main(["--out", str(out)]) == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["verdict"]["conditional"] is True
        # Default sweep: low end derived from the call structure, not hardcoded.
        assert len(payload["scenarios"]) >= 2
        assert payload["scenarios"][0]["stage_share"] < 0.01

    def test_measured_share_below_the_bar_exits_one(self, tmp_path: Path) -> None:
        out = tmp_path / "ceiling.json"
        assert _mod.main(["--observe-share", "0.08", "--out", str(out)]) == 1
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["verdict"]["justified"] is False

    def test_invalid_share_exits_one_without_writing(self, tmp_path: Path) -> None:
        out = tmp_path / "ceiling.json"
        assert _mod.main(["--observe-share", "1.5", "--out", str(out)]) == 1
        assert not out.exists()
