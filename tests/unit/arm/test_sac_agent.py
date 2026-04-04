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
        with patch.dict("sys.modules", {"stable_baselines3": None}), pytest.raises(
            ImportError, match="stable-baselines3"
        ):
            agent.build(env)


class TestSACAgentLoad:
    """Tests for SACAgent.load."""

    def test_load_raises_without_sb3(self, agent: SACAgent) -> None:
        with patch.dict("sys.modules", {"stable_baselines3": None}), pytest.raises(
            ImportError, match="stable-baselines3"
        ):
            agent.load("/fake/path")
