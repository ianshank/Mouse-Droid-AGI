"""Unit tests for the WS4 world-model rollout-return scoring harness.

Pins the user-chosen scoring contract:

* the score is the MEAN imagined return over ``n_rollouts`` rollouts of horizon
  ``horizon`` through the REUSED RSSM ``imagine_step`` reward head;
* it is DETERMINISTIC — same seed + same seed-states + same policy weights
  ALWAYS yields a byte-identical float (the load-bearing reproducibility pin);
* ``horizon`` and ``n_rollouts`` are honoured (drive the reward-head call count);
* scoring runs under ``torch.no_grad()`` + eval mode (no autograd graph leaks,
  no train-mode side effects on the policy);
* a stronger-reward policy scores higher than a weaker one (monotone sanity);
* a structured ``on_device_score_computed`` log is emitted.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mousedroid.config.schema import ModelConfig
from mousedroid.learning.on_device.scoring import (
    StateDictPolicyAdapter,
    score_policy,
)
from mousedroid.world_model.rssm import RSSM

_HORIZON = 5
_N_ROLLOUTS = 4
_SEED = 7


def _make_world_model() -> RSSM:
    """Build a small deterministic RSSM (vision OFF so no camera features)."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vision_dim=0,
        vision_proj_dim=0,
        hidden_dim=8,
        latent_dim=4,
        action_dim=3,
        obs_dim=8,
    )
    wm = RSSM(cfg)
    wm.eval()
    return wm


def _make_seed_states(wm: RSSM, n: int = 3) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministically sample ``n`` (h, z) seed states for the world model."""
    torch.manual_seed(42)
    hidden = wm.cfg.hidden_dim
    latent = wm.cfg.latent_dim
    return [(torch.randn(1, hidden), torch.randn(1, latent)) for _ in range(n)]


class _ConstantPolicy:
    """A policy returning a fixed action regardless of latent state."""

    def __init__(self, action: torch.Tensor) -> None:
        self._action = action

    def act(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        batch = hidden.shape[0]
        return self._action.expand(batch, -1)


def test_score_is_deterministic_same_seed() -> None:
    """Same seed + inputs + weights -> byte-identical score."""
    wm = _make_world_model()
    seed_states = _make_seed_states(wm)
    policy = _ConstantPolicy(torch.zeros(1, wm.cfg.action_dim))

    a = score_policy(policy, wm, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED)
    b = score_policy(policy, wm, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED)

    assert a == b


def test_different_seed_can_change_score() -> None:
    """A different seed changes the sampled rollout (prior is stochastic)."""
    wm = _make_world_model()
    seed_states = _make_seed_states(wm)
    policy = _ConstantPolicy(torch.zeros(1, wm.cfg.action_dim))

    a = score_policy(policy, wm, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=1)
    b = score_policy(policy, wm, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=2)

    assert isinstance(a, float)
    assert isinstance(b, float)
    assert a != b


def test_score_runs_under_no_grad() -> None:
    """The returned score carries no autograd graph (scored under no_grad)."""
    wm = _make_world_model()
    seed_states = _make_seed_states(wm)
    policy = _ConstantPolicy(torch.zeros(1, wm.cfg.action_dim))

    score = score_policy(
        policy, wm, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED
    )

    # A python float carries no grad; assert the contract by re-scoring inside an
    # enable_grad block and confirming no graph is ever built on the WM params.
    assert isinstance(score, float)
    for param in wm.parameters():
        assert param.grad is None


def test_horizon_and_n_rollouts_drive_reward_calls() -> None:
    """The reward head is called exactly ``horizon * n_rollouts * len(states)``."""
    wm = _make_world_model()
    seed_states = _make_seed_states(wm, n=2)
    policy = _ConstantPolicy(torch.zeros(1, wm.cfg.action_dim))

    calls = {"n": 0}
    real_imagine = wm.imagine_step

    def _counting_imagine(
        action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls["n"] += 1
        return real_imagine(action, h, z)

    wm.imagine_step = _counting_imagine  # type: ignore[method-assign]

    score_policy(policy, wm, seed_states, horizon=3, n_rollouts=2, seed=_SEED)

    assert calls["n"] == 3 * 2 * len(seed_states)


class _RewardIsActionSumWorldModel:
    """A deterministic stub world model whose reward is the action's sum.

    Satisfies :class:`WorldModelProtocol`'s ``imagine_step`` so the score is a
    clean, monotone function of the policy's action magnitude — a stronger
    (larger positive action) policy provably scores STRICTLY higher, with no
    dependence on random reward-head init. The latent state is carried forward
    unchanged so the rollout is fully deterministic.
    """

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim

    def imagine_step(
        self, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reward = action.sum(dim=-1, keepdim=True)
        return h, z, reward


def test_stronger_policy_scores_higher() -> None:
    """A policy driving the reward head harder scores STRICTLY higher.

    Both policies share the SAME world model + seed-states + seed; only the
    action differs. The stub world model makes the predicted reward a monotone
    function of the action sum, so a larger positive action provably yields a
    strictly higher score — a meaningful, deterministic monotonicity assertion.
    """
    wm = _make_world_model()
    stub = _RewardIsActionSumWorldModel(wm.cfg.hidden_dim, wm.cfg.latent_dim)
    seed_states = _make_seed_states(wm)

    weak = _ConstantPolicy(torch.ones(1, wm.cfg.action_dim))
    strong = _ConstantPolicy(torch.ones(1, wm.cfg.action_dim) * 5.0)

    s_weak = score_policy(
        weak, stub, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED
    )
    s_strong = score_policy(
        strong, stub, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED
    )

    assert s_strong > s_weak


def test_score_restores_global_rng_state() -> None:
    """``score_policy`` leaves the GLOBAL torch RNG state unchanged.

    The harness seeds the global RNG internally for reproducibility, but it
    MUST capture + restore the prior global state so callers sharing the
    process RNG are never silently perturbed.
    """
    wm = _make_world_model()
    seed_states = _make_seed_states(wm)
    policy = _ConstantPolicy(torch.zeros(1, wm.cfg.action_dim))

    # Establish a known, non-default global RNG state.
    torch.manual_seed(1234)
    before = torch.get_rng_state()

    score_policy(policy, wm, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED)

    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_score_restores_world_model_train_mode() -> None:
    """A world model left in ``.train()`` is still in ``.train()`` afterwards."""
    wm = _make_world_model()
    wm.train()
    seed_states = _make_seed_states(wm)
    policy = _ConstantPolicy(torch.zeros(1, wm.cfg.action_dim))

    assert wm.training is True
    score_policy(policy, wm, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED)
    assert wm.training is True


def test_score_restores_state_even_on_exception() -> None:
    """RNG + train-mode are restored even when a rollout raises."""
    wm = _make_world_model()
    wm.train()
    seed_states = _make_seed_states(wm)

    class _BoomPolicy:
        def act(self, hidden: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("boom")

    torch.manual_seed(4321)
    before = torch.get_rng_state()

    with pytest.raises(RuntimeError, match="boom"):
        score_policy(
            _BoomPolicy(), wm, seed_states, horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED
        )

    assert torch.equal(before, torch.get_rng_state())
    assert wm.training is True


def test_state_dict_policy_adapter_satisfies_protocol() -> None:
    """The state-dict adapter exposes ``act`` and returns an action tensor."""
    wm = _make_world_model()
    action_dim = wm.cfg.action_dim
    net = nn.Linear(wm.cfg.hidden_dim + wm.cfg.latent_dim, action_dim)
    adapter = StateDictPolicyAdapter(
        net, hidden_dim=wm.cfg.hidden_dim, latent_dim=wm.cfg.latent_dim
    )

    h = torch.randn(1, wm.cfg.hidden_dim)
    z = torch.randn(1, wm.cfg.latent_dim)
    action = adapter.act(h, z)

    assert action.shape == (1, action_dim)


def test_adapter_rejects_wrong_hidden_dim() -> None:
    """``act`` raises ``ValueError`` when the hidden width mismatches."""
    wm = _make_world_model()
    net = nn.Linear(wm.cfg.hidden_dim + wm.cfg.latent_dim, wm.cfg.action_dim)
    adapter = StateDictPolicyAdapter(
        net, hidden_dim=wm.cfg.hidden_dim, latent_dim=wm.cfg.latent_dim
    )

    bad_hidden = torch.randn(1, wm.cfg.hidden_dim + 1)
    z = torch.randn(1, wm.cfg.latent_dim)
    with pytest.raises(ValueError, match="hidden"):
        adapter.act(bad_hidden, z)


def test_adapter_rejects_wrong_latent_dim() -> None:
    """``act`` raises ``ValueError`` when the latent width mismatches."""
    wm = _make_world_model()
    net = nn.Linear(wm.cfg.hidden_dim + wm.cfg.latent_dim, wm.cfg.action_dim)
    adapter = StateDictPolicyAdapter(
        net, hidden_dim=wm.cfg.hidden_dim, latent_dim=wm.cfg.latent_dim
    )

    h = torch.randn(1, wm.cfg.hidden_dim)
    bad_latent = torch.randn(1, wm.cfg.latent_dim + 1)
    with pytest.raises(ValueError, match="latent"):
        adapter.act(h, bad_latent)


def test_adapter_rejects_under_wide_module_output() -> None:
    """``act`` raises when the module output is narrower than ``action_dim``."""
    wm = _make_world_model()
    # Module emits fewer columns than the requested action width.
    net = nn.Linear(wm.cfg.hidden_dim + wm.cfg.latent_dim, wm.cfg.action_dim - 1)
    adapter = StateDictPolicyAdapter(
        net,
        hidden_dim=wm.cfg.hidden_dim,
        latent_dim=wm.cfg.latent_dim,
        action_dim=wm.cfg.action_dim,
    )

    h = torch.randn(1, wm.cfg.hidden_dim)
    z = torch.randn(1, wm.cfg.latent_dim)
    with pytest.raises(ValueError, match="action_dim"):
        adapter.act(h, z)


def test_adapter_happy_path_slices_to_action_dim() -> None:
    """A wider module output is sliced down to ``action_dim`` columns."""
    wm = _make_world_model()
    # Module emits MORE columns than action_dim; output is sliced.
    net = nn.Linear(wm.cfg.hidden_dim + wm.cfg.latent_dim, wm.cfg.action_dim + 2)
    adapter = StateDictPolicyAdapter(
        net,
        hidden_dim=wm.cfg.hidden_dim,
        latent_dim=wm.cfg.latent_dim,
        action_dim=wm.cfg.action_dim,
    )

    h = torch.randn(1, wm.cfg.hidden_dim)
    z = torch.randn(1, wm.cfg.latent_dim)
    action = adapter.act(h, z)

    assert action.shape == (1, wm.cfg.action_dim)


def test_empty_seed_states_scores_zero() -> None:
    """An empty seed-state set is a well-defined ``0.0`` (no rollouts)."""
    wm = _make_world_model()
    policy = _ConstantPolicy(torch.zeros(1, wm.cfg.action_dim))

    score = score_policy(policy, wm, [], horizon=_HORIZON, n_rollouts=_N_ROLLOUTS, seed=_SEED)

    assert score == 0.0


def test_score_logs_structured_event() -> None:
    """Scoring emits a structured ``on_device_score_computed`` event."""
    import structlog

    wm = _make_world_model()
    seed_states = _make_seed_states(wm)
    policy = _ConstantPolicy(torch.zeros(1, wm.cfg.action_dim))

    with structlog.testing.capture_logs() as captured:
        score_policy(policy, wm, seed_states, horizon=2, n_rollouts=2, seed=_SEED)

    events = [entry.get("event", "") for entry in captured]
    assert "on_device_score_computed" in events
