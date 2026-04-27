"""End-to-end integration tests for the synthetic data generator.

Exercises both the legacy disabled-DR code path (preserves byte-identical
behaviour) and the new randomized path (DR enabled with a fixed seed). Uses
``mock_hardware=True`` so the orchestrator and sensor manager run without
GPIO, audio, or serial dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from training.data_generator import SyntheticSequenceGenerator

from mousedroid.config.schema import DomainRandomizationConfig, RangeF, Settings


@pytest.fixture
def mock_settings_dr_disabled() -> Settings:
    """Settings with mock hardware and DR disabled — legacy generator path."""
    return Settings(
        mock_hardware=True,
        domain_randomization=DomainRandomizationConfig(enabled=False),
    )


@pytest.fixture
def mock_settings_dr_enabled() -> Settings:
    """Settings with mock hardware and a tame DR envelope."""
    cfg = DomainRandomizationConfig(
        enabled=True,
        feature_noise_std=RangeF(low=0.01, high=0.02),
        ultrasonic_noise_m=RangeF(low=0.005, high=0.01),
        ultrasonic_dropout_prob=RangeF(low=0.0, high=0.0),
    )
    return Settings(mock_hardware=True, domain_randomization=cfg)


class TestGeneratorDisabledPath:
    """The disabled-DR path falls back to legacy ``torch.randn`` actions."""

    def test_generates_expected_sequence_count(
        self, mock_settings_dr_disabled: Settings, tmp_path: Path
    ) -> None:
        gen = SyntheticSequenceGenerator(mock_settings_dr_disabled)
        out = gen.generate_sequences(n_episodes=2, max_steps=4, output_dir=tmp_path)
        episodes = torch.load(out / "sequences.pt", weights_only=False)
        assert len(episodes) == 2
        assert len(episodes[0]) == 4

    def test_seed_param_logged_but_ignored_when_disabled(
        self, mock_settings_dr_disabled: Settings, tmp_path: Path
    ) -> None:
        # Construct with seed; legacy path must still produce torch.randn actions.
        gen = SyntheticSequenceGenerator(mock_settings_dr_disabled, seed=7)
        out = gen.generate_sequences(n_episodes=1, max_steps=2, output_dir=tmp_path)
        assert (out / "sequences.pt").exists()


class TestGeneratorEnabledPath:
    """DR-enabled path samples per-episode params and applies transforms."""

    def test_generates_with_seed(self, mock_settings_dr_enabled: Settings, tmp_path: Path) -> None:
        gen = SyntheticSequenceGenerator(mock_settings_dr_enabled, seed=2024)
        out = gen.generate_sequences(n_episodes=2, max_steps=3, output_dir=tmp_path)
        episodes = torch.load(out / "sequences.pt", weights_only=False)
        assert len(episodes) == 2
        for ep in episodes:
            assert len(ep) == 3
            for step in ep:
                # Vision feature dim is determined by the model config; just
                # assert the tensor is the right rank and dtype.
                assert step["vision"].dtype == torch.float32
                assert step["action"].shape == torch.Size(
                    [mock_settings_dr_enabled.model.action_dim]
                )

    def test_same_seed_yields_same_actions(
        self, mock_settings_dr_enabled: Settings, tmp_path: Path
    ) -> None:
        path_a = tmp_path / "a"
        path_b = tmp_path / "b"
        gen_a = SyntheticSequenceGenerator(mock_settings_dr_enabled, seed=99)
        gen_b = SyntheticSequenceGenerator(mock_settings_dr_enabled, seed=99)
        gen_a.generate_sequences(n_episodes=1, max_steps=4, output_dir=path_a)
        gen_b.generate_sequences(n_episodes=1, max_steps=4, output_dir=path_b)
        ep_a = torch.load(path_a / "sequences.pt", weights_only=False)
        ep_b = torch.load(path_b / "sequences.pt", weights_only=False)
        for s_a, s_b in zip(ep_a[0], ep_b[0], strict=True):
            assert torch.equal(s_a["action"], s_b["action"])

    def test_different_seeds_yield_different_actions(
        self, mock_settings_dr_enabled: Settings, tmp_path: Path
    ) -> None:
        path_a = tmp_path / "a"
        path_b = tmp_path / "b"
        gen_a = SyntheticSequenceGenerator(mock_settings_dr_enabled, seed=1)
        gen_b = SyntheticSequenceGenerator(mock_settings_dr_enabled, seed=2)
        gen_a.generate_sequences(n_episodes=1, max_steps=4, output_dir=path_a)
        gen_b.generate_sequences(n_episodes=1, max_steps=4, output_dir=path_b)
        ep_a = torch.load(path_a / "sequences.pt", weights_only=False)
        ep_b = torch.load(path_b / "sequences.pt", weights_only=False)
        actions_a = torch.stack([s["action"] for s in ep_a[0]])
        actions_b = torch.stack([s["action"] for s in ep_b[0]])
        assert not torch.allclose(actions_a, actions_b)


class TestGeneratorConstructionGuards:
    """Constructor preconditions surface clear ValueError messages."""

    def test_real_hardware_raises(self) -> None:
        cfg = Settings(mock_hardware=False, ultrasonic={"trigger_pin": 23, "echo_pin": 24})
        with pytest.raises(ValueError, match="mock_hardware"):
            SyntheticSequenceGenerator(cfg)
