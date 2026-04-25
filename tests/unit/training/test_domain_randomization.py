"""Tests for the Phase 1 domain randomization sampler and transforms.

The disabled-randomizer path is the backwards-compatibility contract: an
empty :class:`EpisodeParams` is returned and observation transforms are
no-ops, so existing pipelines continue to produce byte-identical artifacts.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from mousedroid.config.schema import DomainRandomizationConfig, RangeF
from mousedroid.training.domain_randomization import (
    DomainRandomizer,
    EpisodeParams,
    apply_feature_noise,
    apply_range_sensor_randomization,
    apply_visual_randomization,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Per-test deterministic RNG."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def cfg() -> DomainRandomizationConfig:
    """Default-enabled domain randomization config."""
    return DomainRandomizationConfig()


# ---------------------------------------------------------------------------
# RangeF
# ---------------------------------------------------------------------------


class TestRangeF:
    """Validation and sampling for the inclusive ``[low, high]`` range model."""

    def test_low_above_high_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RangeF(low=2.0, high=1.0)

    def test_equal_low_high_allowed(self) -> None:
        r = RangeF(low=0.5, high=0.5)
        assert r.low == 0.5
        assert r.high == 0.5

    def test_negative_range_allowed(self) -> None:
        r = RangeF(low=-3.0, high=-1.0)
        assert r.low < r.high


# ---------------------------------------------------------------------------
# DomainRandomizationConfig defaults & overrides
# ---------------------------------------------------------------------------


class TestDomainRandomizationConfig:
    """Schema-level defaults stay backwards compatible."""

    def test_defaults_enabled(self) -> None:
        c = DomainRandomizationConfig()
        assert c.enabled is True
        assert c.brightness.low < c.brightness.high
        assert 0.0 <= c.push_event_prob <= 1.0

    def test_partial_overrides_preserve_defaults(self) -> None:
        c = DomainRandomizationConfig(enabled=False)
        assert c.enabled is False
        assert c.brightness.low == pytest.approx(0.6)
        assert c.brightness.high == pytest.approx(1.4)

    def test_push_event_prob_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DomainRandomizationConfig(push_event_prob=1.5)


# ---------------------------------------------------------------------------
# EpisodeParams
# ---------------------------------------------------------------------------


class TestEpisodeParams:
    """Empty bundles are the disabled-DR sentinel."""

    def test_default_is_empty(self) -> None:
        ep = EpisodeParams()
        assert ep.is_empty
        assert ep.visual == {}
        assert ep.range_sensor == {}

    def test_populated_is_not_empty(self) -> None:
        ep = EpisodeParams(visual={"brightness": 1.0})
        assert not ep.is_empty


# ---------------------------------------------------------------------------
# DomainRandomizer
# ---------------------------------------------------------------------------


class TestDomainRandomizer:
    """Sampler reproducibility and bypass when disabled."""

    def test_disabled_returns_empty_params(self, rng: np.random.Generator) -> None:
        dr = DomainRandomizer(DomainRandomizationConfig(enabled=False))
        assert not dr.enabled
        params = dr.sample(rng)
        assert params.is_empty

    def test_enabled_populates_all_groups(
        self, cfg: DomainRandomizationConfig, rng: np.random.Generator
    ) -> None:
        dr = DomainRandomizer(cfg)
        params = dr.sample(rng)
        for group in (
            params.visual,
            params.camera,
            params.range_sensor,
            params.chassis,
            params.comms,
            params.disturbance,
            params.feature,
        ):
            assert len(group) > 0

    def test_reproducible_with_same_seed(self, cfg: DomainRandomizationConfig) -> None:
        dr = DomainRandomizer(cfg)
        a = dr.sample(np.random.default_rng(seed=123))
        b = dr.sample(np.random.default_rng(seed=123))
        assert dict(a.chassis) == dict(b.chassis)
        assert dict(a.visual) == dict(b.visual)
        assert dict(a.feature) == dict(b.feature)

    def test_diverse_with_different_seeds(self, cfg: DomainRandomizationConfig) -> None:
        dr = DomainRandomizer(cfg)
        a = dr.sample(np.random.default_rng(seed=1))
        b = dr.sample(np.random.default_rng(seed=2))
        assert a.chassis["friction"] != b.chassis["friction"]

    def test_sampled_values_within_configured_ranges(
        self, cfg: DomainRandomizationConfig, rng: np.random.Generator
    ) -> None:
        dr = DomainRandomizer(cfg)
        for _ in range(50):
            p = dr.sample(rng)
            assert cfg.brightness.low <= p.visual["brightness"] <= cfg.brightness.high
            assert cfg.wheel_friction.low <= p.chassis["friction"] <= cfg.wheel_friction.high
            assert cfg.uart_latency_ms.low <= p.comms["uart_latency_ms"] <= cfg.uart_latency_ms.high

    def test_degenerate_range_returns_low_value(self, rng: np.random.Generator) -> None:
        cfg = DomainRandomizationConfig(brightness=RangeF(low=0.5, high=0.5))
        dr = DomainRandomizer(cfg)
        p = dr.sample(rng)
        assert p.visual["brightness"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Visual transform
# ---------------------------------------------------------------------------


class TestVisualRandomization:
    """RGB-frame transforms preserve dtype and clip to valid range."""

    def test_uint8_in_uint8_out(self, rng: np.random.Generator) -> None:
        frame = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        params = {
            "brightness": 1.0,
            "contrast": 1.0,
            "hue_shift_deg": 0.0,
            "gaussian_noise_std": 0.0,
            "motion_blur_px": 0.0,
        }
        out = apply_visual_randomization(frame, params, rng)
        assert out.dtype == np.uint8
        assert out.shape == frame.shape

    def test_float32_in_float32_out(self, rng: np.random.Generator) -> None:
        frame = rng.random((16, 16, 3)).astype(np.float32)
        params = {"brightness": 1.0, "contrast": 1.0, "gaussian_noise_std": 0.0}
        out = apply_visual_randomization(frame, params, rng)
        assert out.dtype == np.float32

    def test_brightness_scales_mean_down(self, rng: np.random.Generator) -> None:
        frame = np.full((8, 8, 3), 0.5, dtype=np.float32)
        params = {"brightness": 0.5, "contrast": 1.0, "gaussian_noise_std": 0.0}
        out = apply_visual_randomization(frame, params, rng)
        assert out.mean() < frame.mean()

    def test_output_clipped_to_unit_interval(self, rng: np.random.Generator) -> None:
        frame = np.full((8, 8, 3), 0.9, dtype=np.float32)
        params = {"brightness": 5.0, "contrast": 1.0, "gaussian_noise_std": 0.0}
        out = apply_visual_randomization(frame, params, rng)
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_noise_changes_output(self, rng: np.random.Generator) -> None:
        frame = np.full((16, 16, 3), 0.5, dtype=np.float32)
        params = {"brightness": 1.0, "contrast": 1.0, "gaussian_noise_std": 0.05}
        out = apply_visual_randomization(frame, params, rng)
        assert not np.allclose(out, frame)


# ---------------------------------------------------------------------------
# Range sensor transform
# ---------------------------------------------------------------------------


class TestRangeSensorRandomization:
    """Additive noise + dropout semantics for HC-SR04 readings."""

    def test_noise_perturbs_reading(self, rng: np.random.Generator) -> None:
        params = {"noise_m": 0.05, "dropout_prob": 0.0}
        readings = np.array(
            [apply_range_sensor_randomization(1.0, params, rng) for _ in range(200)]
        )
        assert abs(readings.mean() - 1.0) < 0.02
        assert readings.std() > 0.0

    def test_full_dropout_returns_nan(self) -> None:
        local_rng = np.random.default_rng(seed=0)
        params = {"noise_m": 0.0, "dropout_prob": 1.0}
        assert np.isnan(apply_range_sensor_randomization(1.5, params, local_rng))

    def test_no_dropout_no_nan(self, rng: np.random.Generator) -> None:
        params = {"noise_m": 0.0, "dropout_prob": 0.0}
        for _ in range(50):
            r = apply_range_sensor_randomization(2.0, params, rng)
            assert not np.isnan(r)

    def test_zero_noise_zero_dropout_is_passthrough(self, rng: np.random.Generator) -> None:
        params = {"noise_m": 0.0, "dropout_prob": 0.0}
        assert apply_range_sensor_randomization(1.234, params, rng) == pytest.approx(1.234)


# ---------------------------------------------------------------------------
# Feature noise transform
# ---------------------------------------------------------------------------


class TestFeatureNoise:
    """Post-CNN feature-vector noise preserves shape and dtype."""

    def test_zero_std_is_passthrough(self, rng: np.random.Generator) -> None:
        features = rng.standard_normal(256, dtype=np.float32)
        out = apply_feature_noise(features, {"noise_std": 0.0}, rng)
        np.testing.assert_array_equal(out, features)

    def test_nonzero_std_perturbs(self, rng: np.random.Generator) -> None:
        features = np.zeros(64, dtype=np.float32)
        out = apply_feature_noise(features, {"noise_std": 0.1}, rng)
        assert out.shape == features.shape
        assert out.dtype == features.dtype
        assert not np.array_equal(out, features)

    def test_batch_shape_preserved(self, rng: np.random.Generator) -> None:
        features = np.zeros((4, 32), dtype=np.float32)
        out = apply_feature_noise(features, {"noise_std": 0.05}, rng)
        assert out.shape == (4, 32)
