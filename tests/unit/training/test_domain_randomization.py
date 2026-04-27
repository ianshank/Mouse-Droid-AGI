from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from mousedroid.config.schema import DomainRandomizationConfig, RangeF
from mousedroid.training.domain_randomization import (
    DomainRandomizer,
    EpisodeParams,
    apply_feature_noise,
    apply_visual_randomization,
)


class TestRangeF:
    def test_low_above_high_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RangeF(low=2.0, high=1.0)

    def test_equal_low_high_allowed(self) -> None:
        cfg = RangeF(low=0.5, high=0.5)
        assert cfg.low == 0.5
        assert cfg.high == 0.5

    def test_negative_range_allowed(self) -> None:
        cfg = RangeF(low=-3.0, high=-1.0)
        assert cfg.low == -3.0
        assert cfg.high == -1.0


class TestDomainRandomizationConfig:
    def test_defaults_enabled(self) -> None:
        cfg = DomainRandomizationConfig()
        assert cfg.enabled is True
        assert cfg.brightness.low < cfg.brightness.high
        assert 0.0 <= cfg.push_event_prob <= 1.0

    def test_partial_overrides_preserve_defaults(self) -> None:
        cfg = DomainRandomizationConfig(enabled=False)
        assert cfg.enabled is False
        assert cfg.brightness.low == 0.6
        assert cfg.brightness.high == 1.4

    def test_push_event_prob_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DomainRandomizationConfig(push_event_prob=1.5)


class TestEpisodeParams:
    def test_default_is_empty(self) -> None:
        params = EpisodeParams()
        assert params.is_empty is True
        assert params.visual == {}
        assert params.feature == {}

    def test_populated_is_not_empty(self) -> None:
        params = EpisodeParams(visual={"brightness": 1.0})
        assert params.is_empty is False


class TestDomainRandomizer:
    def test_disabled_returns_empty_params(self) -> None:
        rng = np.random.default_rng(0)
        params = DomainRandomizer(DomainRandomizationConfig(enabled=False)).sample(rng)
        assert params.is_empty is True

    def test_enabled_populates_all_groups(self) -> None:
        rng = np.random.default_rng(0)
        params = DomainRandomizer(DomainRandomizationConfig()).sample(rng)
        assert params.visual
        assert params.camera
        assert params.chassis
        assert params.comms
        assert params.disturbance
        assert params.feature

    def test_reproducible_with_same_seed(self) -> None:
        cfg = DomainRandomizationConfig()
        params1 = DomainRandomizer(cfg).sample(np.random.default_rng(7))
        params2 = DomainRandomizer(cfg).sample(np.random.default_rng(7))
        assert dict(params1.chassis) == dict(params2.chassis)
        assert dict(params1.visual) == dict(params2.visual)
        assert dict(params1.feature) == dict(params2.feature)

    def test_diverse_with_different_seeds(self) -> None:
        cfg = DomainRandomizationConfig()
        params1 = DomainRandomizer(cfg).sample(np.random.default_rng(1))
        params2 = DomainRandomizer(cfg).sample(np.random.default_rng(2))
        assert params1.chassis["friction"] != params2.chassis["friction"]

    def test_sampled_values_within_configured_ranges(self) -> None:
        cfg = DomainRandomizationConfig()
        randomizer = DomainRandomizer(cfg)
        for seed in range(50):
            params = randomizer.sample(np.random.default_rng(seed))
            assert cfg.brightness.low <= params.visual["brightness"] <= cfg.brightness.high
            assert cfg.wheel_friction.low <= params.chassis["friction"] <= cfg.wheel_friction.high
            assert (
                cfg.uart_latency_ms.low
                <= params.comms["uart_latency_ms"]
                <= cfg.uart_latency_ms.high
            )

    def test_degenerate_range_returns_low_value(self) -> None:
        cfg = DomainRandomizationConfig(brightness=RangeF(low=0.5, high=0.5))
        params = DomainRandomizer(cfg).sample(np.random.default_rng(0))
        assert params.visual["brightness"] == pytest.approx(0.5)


class TestVisualRandomization:
    def test_uint8_in_uint8_out(self) -> None:
        frame = np.full((8, 8, 3), 127, dtype=np.uint8)
        params = EpisodeParams(visual={"brightness": 0.9, "contrast": 1.1})
        randomized = apply_visual_randomization(frame, params, np.random.default_rng(0))
        assert randomized.dtype == np.uint8
        assert randomized.shape == frame.shape

    def test_float32_in_float32_out(self) -> None:
        frame = np.full((8, 8, 3), 0.5, dtype=np.float32)
        params = EpisodeParams(visual={"brightness": 0.9, "contrast": 1.1})
        randomized = apply_visual_randomization(frame, params, np.random.default_rng(0))
        assert randomized.dtype == np.float32

    def test_brightness_scales_mean_down(self) -> None:
        frame = np.full((8, 8, 3), 0.8, dtype=np.float32)
        params = EpisodeParams(visual={"brightness": 0.5})
        randomized = apply_visual_randomization(frame, params, np.random.default_rng(0))
        assert randomized.mean() < frame.mean()

    def test_output_clipped_to_unit_interval(self) -> None:
        frame = np.full((8, 8, 3), 0.8, dtype=np.float32)
        params = EpisodeParams(visual={"brightness": 5.0})
        randomized = apply_visual_randomization(frame, params, np.random.default_rng(0))
        assert randomized.min() >= 0.0
        assert randomized.max() <= 1.0

    def test_noise_changes_output(self) -> None:
        frame = np.full((8, 8, 3), 0.5, dtype=np.float32)
        params = EpisodeParams(visual={"gaussian_noise_std": 0.05})
        randomized = apply_visual_randomization(frame, params, np.random.default_rng(0))
        assert not np.allclose(randomized, frame)
class TestFeatureNoise:
    def test_zero_std_is_passthrough(self) -> None:
        features = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        randomized = apply_feature_noise(features, {"noise_std": 0.0}, np.random.default_rng(0))
        np.testing.assert_array_equal(randomized, features)

    def test_nonzero_std_perturbs(self) -> None:
        features = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        randomized = apply_feature_noise(features, {"noise_std": 0.1}, np.random.default_rng(0))
        assert randomized.shape == features.shape
        assert randomized.dtype == features.dtype
        assert not np.allclose(randomized, features)

    def test_batch_shape_preserved(self) -> None:
        features = np.ones((4, 32), dtype=np.float32)
        randomized = apply_feature_noise(features, {"noise_std": 0.1}, np.random.default_rng(0))
        assert randomized.shape == (4, 32)