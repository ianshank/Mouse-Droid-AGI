"""Unit tests for ``MCTSConfig.action_candidate_strategy`` dispatch.

The legacy ``shared_axis`` set broadcasts one ``linspace`` across every action
axis, so every candidate satisfies ``vx == vy == omega``. That matrix has rank
1: the planner cannot propose "drive straight" (omega=0, vx!=0) or "turn in
place" (vx=0, omega!=0), and its only non-arcing primitive is a full stop.
These tests pin both the legacy shape (unchanged) and the spanning ``per_axis``
alternative that fixes it.
"""

from __future__ import annotations

import torch

from mousedroid.config.schema import MCTSConfig
from mousedroid.constants import DEFAULT_ACTION_DIM, DEFAULT_ACTION_LIMIT
from mousedroid.world_model.mcts import MCTSPlanner

_CPU = torch.device("cpu")


class _StubWorldModel:
    """Minimal ``WorldModelProtocol`` stand-in: identity latent dynamics."""

    def imagine_step(
        self, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return h, z, torch.zeros(1)


def _candidates(strategy: str, *, n: int = 9, action_dim: int = DEFAULT_ACTION_DIM) -> torch.Tensor:
    cfg = MCTSConfig(n_action_candidates=n, action_candidate_strategy=strategy)  # type: ignore[arg-type]
    planner = MCTSPlanner(cfg, _StubWorldModel(), action_dim=action_dim)
    return planner._generate_candidate_actions(_CPU)


def _has_single_axis_move(candidates: torch.Tensor, axis: int) -> bool:
    """True when some candidate moves only along ``axis``."""
    for row in candidates:
        moves = row[axis].abs() > 0
        others = torch.cat((row[:axis], row[axis + 1 :])).abs().max() == 0
        if bool(moves and others):
            return True
    return False


def test_shared_axis_is_the_default_strategy() -> None:
    """Existing deployments keep the pre-existing candidate set."""
    assert MCTSConfig().action_candidate_strategy == "shared_axis"


def test_shared_axis_matrix_is_rank_one() -> None:
    """Characterises the defect: every axis carries the same value."""
    candidates = _candidates("shared_axis")
    assert candidates.shape == (9, DEFAULT_ACTION_DIM)
    assert int(torch.linalg.matrix_rank(candidates)) == 1
    for row in candidates:
        assert torch.allclose(row, row[0].expand_as(row))


def test_shared_axis_cannot_drive_straight_or_turn_in_place() -> None:
    """The two primitives a differential-drive rover most needs are absent."""
    candidates = _candidates("shared_axis")
    assert not _has_single_axis_move(candidates, 0)
    assert not _has_single_axis_move(candidates, DEFAULT_ACTION_DIM - 1)


def test_per_axis_spans_the_action_space() -> None:
    """The spanning set reaches full rank for the default candidate count."""
    candidates = _candidates("per_axis")
    assert candidates.shape == (9, DEFAULT_ACTION_DIM)
    assert int(torch.linalg.matrix_rank(candidates)) == DEFAULT_ACTION_DIM


def test_per_axis_offers_straight_turn_and_stop() -> None:
    """Each motion primitive the rover needs is individually reachable."""
    candidates = _candidates("per_axis")
    assert _has_single_axis_move(candidates, 0), "no drive-straight candidate"
    assert _has_single_axis_move(candidates, DEFAULT_ACTION_DIM - 1), "no turn-in-place candidate"
    assert bool((candidates.abs().sum(dim=1) == 0).any()), "no full-stop candidate"


def test_per_axis_axis_primitives_are_at_full_scale() -> None:
    """The axis moves must be at +/-limit, not merely non-zero.

    ``_execute_action`` multiplies the action by ``esp32.max_velocity_mps``, so
    a primitive at a fraction of full scale would leave the rover unable to
    command full speed on any single axis — passing every rank and
    reachability pin while being as useless as the rank-1 set it replaces.
    """
    candidates = _candidates("per_axis")
    for axis in range(DEFAULT_ACTION_DIM):
        for sign in (1.0, -1.0):
            expected = torch.zeros(DEFAULT_ACTION_DIM)
            expected[axis] = sign * DEFAULT_ACTION_LIMIT
            assert bool((candidates == expected).all(dim=1).any()), (
                f"missing full-scale primitive axis={axis} sign={sign:+.0f}"
            )


def test_per_axis_interleaves_positive_and_negative_directions() -> None:
    """Truncation must drop whole axes, not every negative direction.

    Pins the ordering claim in ``_per_axis_candidates``: rows after the stop
    action run +axis0, -axis0, +axis1, -axis1, ... A grouped ``[+all, -all]``
    layout would strip reverse motion first under truncation.
    """
    candidates = _candidates("per_axis")
    for axis in range(DEFAULT_ACTION_DIM):
        positive = candidates[1 + 2 * axis]
        negative = candidates[2 + 2 * axis]
        assert float(positive[axis]) == DEFAULT_ACTION_LIMIT
        assert float(negative[axis]) == -DEFAULT_ACTION_LIMIT


def test_per_axis_has_no_duplicate_candidates() -> None:
    """A duplicate would waste a slot and double-weight that action in the tree.

    The fill is an irrational rotation evaluated from step 1, so it cannot hit
    the cube centre or the axis primitives. This asserts that invariant rather
    than defending it at runtime.
    """
    for n in (8, 9, 13, 20, 33):
        candidates = _candidates("per_axis", n=n)
        assert candidates.shape[0] == n
        unique = torch.unique(candidates, dim=0)
        assert unique.shape[0] == n, f"duplicate candidate at n={n}"


def test_per_axis_emits_no_signed_negative_zero() -> None:
    """``-axes`` produces -0.0; those must not reach the serial codec.

    ``build_velocity`` would put a signed negative zero on the wire that the
    legacy path never produced.
    """
    candidates = _candidates("per_axis", n=24)
    assert not bool((torch.signbit(candidates) & (candidates == 0)).any())


def test_per_axis_is_deterministic_across_instances() -> None:
    """The deterministic fill keeps planning reproducible run to run."""
    first = _candidates("per_axis")
    second = _candidates("per_axis")
    assert torch.equal(first, second)


def test_per_axis_stays_within_the_action_limit() -> None:
    """Candidates never exceed the normalised action bound."""
    candidates = _candidates("per_axis", n=24)
    assert bool((candidates.abs() <= DEFAULT_ACTION_LIMIT + 1e-6).all())


def test_per_axis_keeps_stop_when_candidates_are_scarce() -> None:
    """Truncation drops fill and far axes first — never the stop action."""
    candidates = _candidates("per_axis", n=2)
    assert candidates.shape == (2, DEFAULT_ACTION_DIM)
    assert bool((candidates[0].abs().sum() == 0).item())


def test_per_axis_returns_contiguous_storage() -> None:
    """Unlike the legacy ``expand`` view, children do not alias one buffer."""
    assert _candidates("per_axis").is_contiguous()


def test_per_axis_is_dimension_agnostic() -> None:
    """Nothing in the spanning set hardcodes a 3-axis action space."""
    for action_dim in (1, 2, 4, 6):
        candidates = _candidates("per_axis", n=4 * action_dim, action_dim=action_dim)
        assert candidates.shape == (4 * action_dim, action_dim)
        assert int(torch.linalg.matrix_rank(candidates)) == action_dim


def test_plan_returns_a_per_axis_candidate() -> None:
    """End-to-end: the strategy reaches the action ``plan`` actually returns."""
    cfg = MCTSConfig(n_simulations_base=4, action_candidate_strategy="per_axis")
    planner = MCTSPlanner(cfg, _StubWorldModel(), action_dim=DEFAULT_ACTION_DIM)
    h = torch.zeros(1, 8)
    z = torch.zeros(1, 4)
    action = planner.plan(h, z)
    assert action.shape == (1, DEFAULT_ACTION_DIM)
    expected = planner._generate_candidate_actions(_CPU)
    assert bool((expected == action).all(dim=1).any()), "returned action is not a candidate"


def test_warm_start_tuning_uses_the_configured_candidate_strategy() -> None:
    """UCB tuning must plan over the candidate set the rover will deploy.

    ``ucb_c`` trades exploration against a specific branching factor, so tuning
    on the legacy diagonal and deploying ``per_axis`` would ship a constant
    fitted to a different action space. ``tune_ucb`` rebuilds ``MCTSConfig``
    field by field, which is exactly where a new field gets silently dropped.
    """
    import inspect

    from training import warmstart_policy

    source = inspect.getsource(warmstart_policy.tune_ucb)
    assert "action_candidate_strategy=base_cfg.action_candidate_strategy" in source, (
        "tune_ucb rebuilds MCTSConfig without propagating action_candidate_strategy"
    )
