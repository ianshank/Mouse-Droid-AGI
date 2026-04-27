"""Unit tests for :class:`RealSimMixer`."""

from __future__ import annotations

import math

import pytest

from mousedroid.training.replay import MixerConfig, RealSimMixer


def _ramped_run(target: float, n: int, ramp_steps: int = 1, seed: int = 0) -> RealSimMixer[str]:
    """Run the mixer over ``n`` steps with sentinel sources, return the mixer."""
    sim = (f"sim-{i}" for i in range(n * 4))
    real = (f"real-{i}" for i in range(n * 4))
    cfg = MixerConfig(alpha_target=target, alpha_ramp_steps=ramp_steps, seed=seed)
    mixer: RealSimMixer[str] = RealSimMixer(sim, real, cfg)
    for consumed, _ in enumerate(mixer, start=1):
        if consumed >= n:
            break
    return mixer


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_realized_alpha_matches_target_within_tolerance(alpha: float) -> None:
    n = 10_000
    mixer = _ramped_run(target=alpha, n=n, ramp_steps=1, seed=42)
    realized = mixer.stats["realized_alpha"]
    # 1% absolute tolerance over 10k draws is safely outside binomial noise.
    assert math.isclose(realized, alpha, abs_tol=0.015), f"target={alpha} realized={realized}"


def test_alpha_zero_produces_no_real_draws() -> None:
    mixer = _ramped_run(target=0.0, n=500, seed=7)
    assert mixer.stats["real_drawn"] == 0
    assert mixer.stats["sim_drawn"] == 500


def test_seed_determinism_byte_identical() -> None:
    a = _ramped_run(target=0.4, n=200, seed=123)
    b = _ramped_run(target=0.4, n=200, seed=123)
    assert a.stats == b.stats


def test_different_seeds_diverge() -> None:
    a = _ramped_run(target=0.4, n=200, seed=1)
    b = _ramped_run(target=0.4, n=200, seed=2)
    assert a.stats["real_drawn"] != b.stats["real_drawn"]


def test_ramp_is_monotone_non_decreasing() -> None:
    sim = iter([f"s-{i}" for i in range(2000)])
    real = iter([f"r-{i}" for i in range(2000)])
    cfg = MixerConfig(alpha_target=0.5, alpha_ramp_steps=1000, seed=0)
    mixer: RealSimMixer[str] = RealSimMixer(sim, real, cfg)
    last = -1.0
    for consumed, _ in enumerate(mixer, start=1):
        cur = mixer.stats["current_alpha"]
        assert cur + 1e-9 >= last
        last = cur
        if consumed >= 1500:
            break
    # After ramp completes, alpha should be at target.
    assert math.isclose(mixer.stats["current_alpha"], 0.5)


def test_real_exhaustion_falls_back_to_sim() -> None:
    sim = iter([f"s-{i}" for i in range(100)])
    real = iter(["only-real"])
    cfg = MixerConfig(alpha_target=1.0, alpha_ramp_steps=1, seed=0)
    mixer: RealSimMixer[str] = RealSimMixer(sim, real, cfg)
    items = list(mixer)
    # Step 0 emits with alpha=0 -> sim; step 1 alpha ramps to 1.0 -> real;
    # remaining steps fall back to sim once real is exhausted.
    assert "only-real" in items
    assert mixer.stats["real_exhausted"] >= 1
    assert mixer.stats["real_drawn"] == 1
    assert mixer.stats["sim_drawn"] == 100


def test_both_exhausted_stops_iteration() -> None:
    sim = iter(["s"])
    real = iter(["r"])
    cfg = MixerConfig(alpha_target=0.5, alpha_ramp_steps=1, seed=0)
    mixer: RealSimMixer[str] = RealSimMixer(sim, real, cfg)
    items = list(mixer)
    assert sorted(items) == ["r", "s"]


def test_invalid_alpha_target_rejected() -> None:
    with pytest.raises(ValueError, match="alpha_target"):
        MixerConfig(alpha_target=1.5)
    with pytest.raises(ValueError, match="alpha_target"):
        MixerConfig(alpha_target=-0.1)


def test_invalid_ramp_steps_rejected() -> None:
    with pytest.raises(ValueError, match="alpha_ramp_steps"):
        MixerConfig(alpha_ramp_steps=0)
