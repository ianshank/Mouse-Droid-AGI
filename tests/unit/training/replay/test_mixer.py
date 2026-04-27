"""Unit tests for ``mousedroid.training.replay.mixer.EpisodeMixer``."""

from __future__ import annotations

import numpy as np
import pytest

from mousedroid.training.replay.mixer import EpisodeMixer


def _make_mixer(
    *,
    alpha_target: float = 0.3,
    ramp: int = 0,
    seed: int = 42,
    has_real: bool = True,
) -> EpisodeMixer:
    return EpisodeMixer(
        alpha_target=alpha_target,
        alpha_ramp_steps=ramp,
        rng=np.random.default_rng(seed),
        has_real_pool=has_real,
    )


def test_alpha_target_outside_range_rejected() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="alpha_target must be in"):
        EpisodeMixer(alpha_target=1.1, alpha_ramp_steps=0, rng=rng, has_real_pool=True)
    with pytest.raises(ValueError, match="alpha_target must be in"):
        EpisodeMixer(alpha_target=-0.1, alpha_ramp_steps=0, rng=rng, has_real_pool=True)


def test_negative_ramp_steps_rejected() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="alpha_ramp_steps must be >= 0"):
        EpisodeMixer(alpha_target=0.5, alpha_ramp_steps=-1, rng=rng, has_real_pool=True)


def test_no_real_pool_always_returns_sim() -> None:
    mixer = _make_mixer(alpha_target=1.0, has_real=False)
    for step in range(50):
        assert mixer.draw(step) == "sim"
    assert mixer.stats.real_draws == 0
    assert mixer.stats.sim_draws == 50


def test_alpha_target_zero_always_returns_sim() -> None:
    mixer = _make_mixer(alpha_target=0.0)
    for step in range(50):
        assert mixer.draw(step) == "sim"
    assert mixer.alpha_at(0) == 0.0
    assert mixer.alpha_at(1000) == 0.0


def test_alpha_target_one_always_returns_real() -> None:
    mixer = _make_mixer(alpha_target=1.0)
    for step in range(50):
        assert mixer.draw(step) == "real"


def test_realized_ratio_within_one_percent_of_target() -> None:
    mixer = _make_mixer(alpha_target=0.3, ramp=0, seed=12345)
    n = 10_000
    for _ in range(n):
        mixer.draw(step=0)
    assert abs(mixer.stats.realized_ratio - 0.3) < 0.01


def test_alpha_ramp_linear_interpolation() -> None:
    mixer = _make_mixer(alpha_target=0.4, ramp=100)
    assert mixer.alpha_at(0) == pytest.approx(0.0)
    assert mixer.alpha_at(50) == pytest.approx(0.2)
    assert mixer.alpha_at(100) == pytest.approx(0.4)
    # Past the ramp window the alpha clamps to alpha_target.
    assert mixer.alpha_at(1_000_000) == pytest.approx(0.4)


def test_negative_step_rejected() -> None:
    mixer = _make_mixer()
    with pytest.raises(ValueError, match="step must be >= 0"):
        mixer.alpha_at(-1)


def test_generator_determinism_same_seed_same_draws() -> None:
    a = _make_mixer(alpha_target=0.5, seed=7)
    b = _make_mixer(alpha_target=0.5, seed=7)
    seq_a = [a.draw(0) for _ in range(200)]
    seq_b = [b.draw(0) for _ in range(200)]
    assert seq_a == seq_b


def test_generator_determinism_different_seed_diverges() -> None:
    a = _make_mixer(alpha_target=0.5, seed=7)
    b = _make_mixer(alpha_target=0.5, seed=8)
    seq_a = [a.draw(0) for _ in range(200)]
    seq_b = [b.draw(0) for _ in range(200)]
    assert seq_a != seq_b


def test_draw_batch_matches_individual_draws_in_distribution() -> None:
    mixer = _make_mixer(alpha_target=0.25, seed=2024)
    batch = mixer.draw_batch(step=0, batch_size=10_000)
    realized = sum(1 for s in batch if s == "real") / len(batch)
    assert abs(realized - 0.25) < 0.015


def test_draw_batch_invalid_size_rejected() -> None:
    mixer = _make_mixer()
    with pytest.raises(ValueError, match="batch_size must be positive"):
        mixer.draw_batch(step=0, batch_size=0)


def test_draw_batch_alpha_zero_short_circuits() -> None:
    mixer = _make_mixer(alpha_target=0.0)
    out = mixer.draw_batch(step=0, batch_size=64)
    assert all(s == "sim" for s in out)
    assert mixer.stats.real_draws == 0
    assert mixer.stats.sim_draws == 64


def test_draw_batch_alpha_one_short_circuits() -> None:
    mixer = _make_mixer(alpha_target=1.0)
    out = mixer.draw_batch(step=0, batch_size=64)
    assert all(s == "real" for s in out)
    assert mixer.stats.real_draws == 64


def test_stats_accumulate_across_calls() -> None:
    mixer = _make_mixer(alpha_target=1.0)
    for _ in range(5):
        mixer.draw(step=0)
    mixer.draw_batch(step=0, batch_size=10)
    assert mixer.stats.total == 15
    assert mixer.stats.realized_ratio == pytest.approx(1.0)
