from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mousedroid.config.schema import Settings
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from training.data_generator import SyntheticSequenceGenerator


class _FakeSensorManager:
    async def read_all(self) -> MouseDroidObservationBundle:
        return MouseDroidObservationBundle(
            _vision_features=np.linspace(0.0, 1.0, 256, dtype=np.float32),
            _distance_m=1.25,
            _motor_state=np.array([0.1, 0.0, 0.2, 12.0], dtype=np.float32),
            _lidar_features=np.linspace(0.0, 1.0, 36, dtype=np.float32),
            _valid_mask=np.ones(5, dtype=np.float32),
        )


class _FakeOrchestrator:
    def __init__(self) -> None:
        self._sensor_manager = _FakeSensorManager()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _make_cfg(*, enabled: bool, tmp_path: Path, mock_hardware: bool = True) -> Settings:
    payload = {
        "mock_hardware": mock_hardware,
        "lidar": {"enabled": True, "feature_dim": 36},
        "model": {
            "lidar_dim": 36,
            "lidar_proj_dim": 32,
        },
        "training": {
            "data_dir": str(tmp_path / "data"),
            "n_episodes": 2,
            "sequence_length": 4,
            "generation": {"log_every_n_episodes": 1},
        },
        "domain_randomization": {"enabled": enabled},
    }
    if not mock_hardware:
        payload["ultrasonic"] = {"trigger_pin": 23, "echo_pin": 24}
    return Settings.model_validate(payload)


def _load_sequences(path: Path) -> list[list[dict[str, torch.Tensor]]]:
    return torch.load(path / "sequences.pt")


class TestGeneratorDisabledPath:
    @patch("training.data_generator.build_orchestrator", side_effect=lambda cfg: _FakeOrchestrator())
    def test_generates_expected_sequence_count(self, _mock_build: object, tmp_path: Path) -> None:
        cfg = _make_cfg(enabled=False, tmp_path=tmp_path)

        output_dir = SyntheticSequenceGenerator(cfg).generate_sequences(
            n_episodes=2,
            max_steps=4,
            output_dir=tmp_path / "disabled",
        )

        sequences = _load_sequences(output_dir)
        assert len(sequences) == 2
        assert all(len(episode) == 4 for episode in sequences)

    @patch("training.data_generator.build_orchestrator", side_effect=lambda cfg: _FakeOrchestrator())
    def test_seed_param_logged_but_ignored_when_disabled(
        self,
        _mock_build: object,
        tmp_path: Path,
    ) -> None:
        cfg = _make_cfg(enabled=False, tmp_path=tmp_path)

        output_dir = SyntheticSequenceGenerator(cfg, seed=7).generate_sequences(
            n_episodes=1,
            max_steps=2,
            output_dir=tmp_path / "disabled-seeded",
        )

        assert (output_dir / "sequences.pt").exists()


class TestGeneratorEnabledPath:
    @patch("training.data_generator.build_orchestrator", side_effect=lambda cfg: _FakeOrchestrator())
    def test_generates_with_seed(self, _mock_build: object, tmp_path: Path) -> None:
        cfg = _make_cfg(enabled=True, tmp_path=tmp_path)

        output_dir = SyntheticSequenceGenerator(cfg, seed=2024).generate_sequences(
            n_episodes=2,
            max_steps=3,
            output_dir=tmp_path / "enabled",
        )

        sequences = _load_sequences(output_dir)
        assert len(sequences) == 2
        assert all(len(episode) == 3 for episode in sequences)
        for episode in sequences:
            for step in episode:
                assert step["vision"].dtype == torch.float32
                assert list(step["action"].shape) == [cfg.model.action_dim]

    @patch("training.data_generator.build_orchestrator", side_effect=lambda cfg: _FakeOrchestrator())
    def test_same_seed_yields_same_actions(self, _mock_build: object, tmp_path: Path) -> None:
        cfg = _make_cfg(enabled=True, tmp_path=tmp_path)

        output_dir_a = SyntheticSequenceGenerator(cfg, seed=99).generate_sequences(
            n_episodes=2,
            max_steps=3,
            output_dir=tmp_path / "same-a",
        )
        output_dir_b = SyntheticSequenceGenerator(cfg, seed=99).generate_sequences(
            n_episodes=2,
            max_steps=3,
            output_dir=tmp_path / "same-b",
        )

        sequences_a = _load_sequences(output_dir_a)
        sequences_b = _load_sequences(output_dir_b)
        for episode_a, episode_b in zip(sequences_a, sequences_b, strict=True):
            for step_a, step_b in zip(episode_a, episode_b, strict=True):
                assert torch.allclose(step_a["action"], step_b["action"])

    @patch("training.data_generator.build_orchestrator", side_effect=lambda cfg: _FakeOrchestrator())
    def test_different_seeds_yield_different_actions(
        self,
        _mock_build: object,
        tmp_path: Path,
    ) -> None:
        cfg = _make_cfg(enabled=True, tmp_path=tmp_path)

        output_dir_a = SyntheticSequenceGenerator(cfg, seed=1).generate_sequences(
            n_episodes=2,
            max_steps=3,
            output_dir=tmp_path / "diff-a",
        )
        output_dir_b = SyntheticSequenceGenerator(cfg, seed=2).generate_sequences(
            n_episodes=2,
            max_steps=3,
            output_dir=tmp_path / "diff-b",
        )

        sequences_a = _load_sequences(output_dir_a)
        sequences_b = _load_sequences(output_dir_b)
        stacked_a = torch.stack([step["action"] for episode in sequences_a for step in episode])
        stacked_b = torch.stack([step["action"] for episode in sequences_b for step in episode])
        assert not torch.allclose(stacked_a, stacked_b)


class TestGeneratorConstructionGuards:
    def test_real_hardware_raises(self, tmp_path: Path) -> None:
        cfg = _make_cfg(enabled=True, tmp_path=tmp_path, mock_hardware=False)
        with pytest.raises(ValueError, match="mock_hardware"):
            SyntheticSequenceGenerator(cfg)