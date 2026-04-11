"""Tests for mousedroid.arm.control.sac_agent — SAC+HER agent wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mousedroid.arm.control.sac_agent import SACAgent
from mousedroid.config.schema import ArmTrainingConfig


@pytest.fixture
def training_cfg() -> ArmTrainingConfig:
    """Create a default ArmTrainingConfig."""
    return ArmTrainingConfig()


@pytest.fixture
def agent(training_cfg: ArmTrainingConfig) -> SACAgent:
    """Create a SACAgent with default config."""
    return SACAgent(training_cfg)


class TestSACAgentInit:
    """Tests for SACAgent initialisation."""

    def test_creates_agent(self, agent: SACAgent) -> None:
        assert agent is not None
        assert not agent.is_trained

    def test_model_not_built_initially(self, agent: SACAgent) -> None:
        assert agent._model is None


class TestSACAgentPredict:
    """Tests for SACAgent.predict."""

    def test_predict_raises_without_build(self, agent: SACAgent) -> None:
        obs = {"observation": np.zeros(3), "desired_goal": np.zeros(3)}
        with pytest.raises(RuntimeError, match="not built"):
            agent.predict(obs)


class TestSACAgentTrain:
    """Tests for SACAgent.train."""

    def test_train_raises_without_build(self, agent: SACAgent) -> None:
        with pytest.raises(RuntimeError, match="not built"):
            agent.train()


class TestSACAgentSave:
    """Tests for SACAgent.save."""

    def test_save_raises_without_build(self, agent: SACAgent) -> None:
        with pytest.raises(RuntimeError, match="not built"):
            agent.save()


class TestSACAgentBuild:
    """Tests for SACAgent.build."""

    def test_build_raises_without_sb3(self, agent: SACAgent) -> None:
        env = MagicMock()
        with (
            patch.dict("sys.modules", {"stable_baselines3": None}),
            pytest.raises(ImportError, match="stable-baselines3"),
        ):
            agent.build(env)


class TestSACAgentBuildWithMock:
    """Tests for SACAgent.build with mocked stable-baselines3."""

    def test_build_creates_model(self, agent: SACAgent) -> None:
        env = MagicMock()
        mock_sac_cls = MagicMock()
        mock_her_cls = MagicMock()
        mock_sb3 = MagicMock()
        mock_sb3.SAC = mock_sac_cls
        mock_sb3.HerReplayBuffer = mock_her_cls

        with patch.dict("sys.modules", {"stable_baselines3": mock_sb3}):
            agent.build(env)

        mock_sac_cls.assert_called_once()
        assert agent._model is not None


class TestSACAgentTrainWithMock:
    """Tests for SACAgent.train with mocked model."""

    def test_train_calls_learn(self, agent: SACAgent) -> None:
        mock_model = MagicMock()
        agent._model = mock_model

        agent.train(total_timesteps=100)

        mock_model.learn.assert_called_once_with(total_timesteps=100)
        assert agent.is_trained

    def test_train_uses_config_timesteps(self, agent: SACAgent) -> None:
        mock_model = MagicMock()
        agent._model = mock_model

        agent.train()

        mock_model.learn.assert_called_once_with(
            total_timesteps=agent._cfg.total_timesteps,
        )


class TestSACAgentPredictWithMock:
    """Tests for SACAgent.predict with mocked model."""

    def test_predict_returns_action(self, agent: SACAgent) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.1, 0.2, 0.3]), None)
        agent._model = mock_model

        obs = {"observation": np.zeros(3), "desired_goal": np.zeros(3)}
        action = agent.predict(obs)

        mock_model.predict.assert_called_once_with(obs, deterministic=True)
        assert action.dtype == np.float64
        np.testing.assert_allclose(action, [0.1, 0.2, 0.3])


class TestSACAgentSaveWithMock:
    """Tests for SACAgent.save with mocked model."""

    def test_save_writes_checkpoint(self, agent: SACAgent, tmp_path) -> None:
        mock_model = MagicMock()
        agent._model = mock_model

        result = agent.save(str(tmp_path))

        mock_model.save.assert_called_once()
        assert result.parent.exists()

    def test_save_uses_default_path(self, agent: SACAgent) -> None:
        mock_model = MagicMock()
        agent._model = mock_model

        result = agent.save()

        assert "sac_her_checkpoint" in str(result)


class TestSACAgentLoad:
    """Tests for SACAgent.load."""

    def test_load_raises_without_sb3(self, agent: SACAgent) -> None:
        with (
            patch.dict("sys.modules", {"stable_baselines3": None}),
            pytest.raises(ImportError, match="stable-baselines3"),
        ):
            agent.load("/fake/path")

    def test_load_with_mocked_sb3(self, agent: SACAgent) -> None:
        mock_loaded_model = MagicMock()
        mock_sac_cls = MagicMock()
        mock_sac_cls.load.return_value = mock_loaded_model
        mock_sb3 = MagicMock()
        mock_sb3.SAC = mock_sac_cls

        with patch.dict("sys.modules", {"stable_baselines3": mock_sb3}):
            agent.load("/fake/path")

        mock_sac_cls.load.assert_called_once_with("/fake/path")
        assert agent._model is mock_loaded_model
        assert agent.is_trained
