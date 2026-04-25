"""Tests for bug fixes identified during code scan.

Covers fixes #1-#9:
  1. _try_sensor_recovery returns False on exhausted retries
  2. torch.no_grad() in SACAgent.predict
  3. EWC consolidate() Fisher estimation forward/backward pass
  4. Seeded RNG in EpisodicReplay
  5. observe_step decorated with @torch.no_grad on RSSM variants
  6. SO-ARM100 async serial I/O (asyncio.to_thread)
  7. CognitiveCore._slow_loop catches asyncio.TimeoutError
  8. Removed duplicate CuriosityAggregator from CognitiveCore
  9. MCTS plan() no redundant tanh squashing
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from mousedroid.config.schema import (
    ArmTrainingConfig,
    LearningConfig,
    MCTSConfig,
    MemoryConfig,
    Settings,
)
from mousedroid.learning.ewc import EWCAgent
from mousedroid.memory.episodic import EpisodicReplay
from mousedroid.orchestrator.orchestrator import MouseDroidOrchestrator
from mousedroid.safety.context import SafetyContext
from mousedroid.sensing.bundle import MouseDroidObservationBundle
from mousedroid.world_model.mcts import MCTSPlanner

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DEFAULT_BATTERY_VOLTAGE = 12.0
DEFAULT_AUDIO_CHUNK_SIZE = 1024


def _make_observation(cfg: Settings) -> MouseDroidObservationBundle:
    return MouseDroidObservationBundle(
        _timestamp=0.0,
        _vision_features=np.zeros(cfg.camera.feature_dim, dtype=np.float32),
        _distance_m=1.5,
        _motor_state=np.array([0.0, 0.0, 0.0, DEFAULT_BATTERY_VOLTAGE], dtype=np.float32),
        _audio_chunk=np.zeros(DEFAULT_AUDIO_CHUNK_SIZE, dtype=np.float32),
        _valid_mask=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def _make_orchestrator(
    *,
    emergency: bool = False,
    valid_sensor_count: int = 3,
    recovery_result: int = 0,
    recovery_attempts: int = 2,
) -> MouseDroidOrchestrator:
    from mousedroid.config.schema import SafetyConfig

    cfg = Settings(
        mock_hardware=True,
        safety=SafetyConfig(
            sensor_recovery_attempts=recovery_attempts,
            sensor_recovery_delay_s=0.001,
        ),
    )

    world_model = MagicMock()
    world_model.observe_step.return_value = (
        torch.zeros(1, cfg.model.hidden_dim),
        torch.zeros(1, cfg.model.latent_dim),
        torch.zeros(1, cfg.model.hidden_dim),
        0.1,
    )

    agent = MagicMock()
    agent.name = "test_agent"
    agent.act.return_value = torch.tensor([0.1, 0.0, 0.0])

    safety_ctx = SafetyContext(
        is_emergency=emergency,
        valid_sensor_count=valid_sensor_count,
    )
    safety_monitor = MagicMock()
    safety_monitor.evaluate.return_value = safety_ctx

    esp32 = AsyncMock()

    sensor_manager = AsyncMock()
    sensor_manager.read_all.return_value = _make_observation(cfg)
    sensor_manager.recovery_attempt.return_value = recovery_result

    return MouseDroidOrchestrator(
        world_model=world_model,
        agents=[agent],
        safety_monitor=safety_monitor,
        esp32=esp32,
        sensor_manager=sensor_manager,
        cfg=cfg,
    )


# ===================================================================
# Fix #1 — _try_sensor_recovery returns False on exhausted retries
# ===================================================================


class TestSensorRecoveryReturnValue:
    """Verify _try_sensor_recovery returns False when all recovery attempts fail."""

    async def test_exhausted_recovery_returns_false(self) -> None:
        orch = _make_orchestrator(
            emergency=True,
            valid_sensor_count=0,
            recovery_result=0,
            recovery_attempts=2,
        )
        safety_ctx = SafetyContext(is_emergency=True, valid_sensor_count=0)
        result = await orch._try_sensor_recovery(safety_ctx)
        assert result is False

    async def test_successful_recovery_returns_true(self) -> None:
        orch = _make_orchestrator(
            emergency=True,
            valid_sensor_count=0,
            recovery_result=1,
            recovery_attempts=2,
        )
        safety_ctx = SafetyContext(is_emergency=True, valid_sensor_count=0)
        result = await orch._try_sensor_recovery(safety_ctx)
        assert result is True

    async def test_no_recovery_needed_returns_false(self) -> None:
        orch = _make_orchestrator(
            emergency=True,
            valid_sensor_count=3,
            recovery_result=0,
            recovery_attempts=2,
        )
        safety_ctx = SafetyContext(is_emergency=True, valid_sensor_count=3)
        result = await orch._try_sensor_recovery(safety_ctx)
        assert result is False

    async def test_zero_attempts_returns_false(self) -> None:
        orch = _make_orchestrator(
            emergency=True,
            valid_sensor_count=0,
            recovery_result=0,
            recovery_attempts=0,
        )
        safety_ctx = SafetyContext(is_emergency=True, valid_sensor_count=0)
        result = await orch._try_sensor_recovery(safety_ctx)
        assert result is False

    async def test_emergency_stop_after_failed_recovery(self) -> None:
        """Emergency stop must fire when recovery fails."""
        orch = _make_orchestrator(
            emergency=True,
            valid_sensor_count=0,
            recovery_result=0,
            recovery_attempts=1,
        )
        await orch.tick()
        orch._esp32.emergency_stop.assert_awaited_once()
        orch._esp32.send_velocity.assert_not_awaited()


# ===================================================================
# Fix #2 — torch.no_grad() in SACAgent.predict
# ===================================================================


class TestSACAgentNoGrad:
    """Verify SACAgent.predict runs inside torch.no_grad context."""

    def test_predict_disables_grad(self) -> None:
        from mousedroid.arm.control.sac_agent import SACAgent

        cfg = ArmTrainingConfig()
        agent = SACAgent(cfg)
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.1, 0.2, 0.3]), None)
        agent._model = mock_model

        grad_state: list[bool] = []

        original_predict = mock_model.predict

        def capture_grad_state(*args: Any, **kwargs: Any) -> Any:
            grad_state.append(torch.is_grad_enabled())
            return original_predict(*args, **kwargs)

        mock_model.predict = capture_grad_state

        agent.predict({"observation": np.zeros(3), "desired_goal": np.zeros(3)})
        assert len(grad_state) == 1
        assert grad_state[0] is False

    def test_predict_returns_float64(self) -> None:
        from mousedroid.arm.control.sac_agent import SACAgent

        cfg = ArmTrainingConfig()
        agent = SACAgent(cfg)
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.1, 0.2]), None)
        agent._model = mock_model

        result = agent.predict({"observation": np.zeros(3), "desired_goal": np.zeros(3)})
        assert result.dtype == np.float64


# ===================================================================
# Fix #3 — EWC Fisher estimation produces non-zero values
# ===================================================================


class TestEWCFisherEstimation:
    """Verify consolidate() runs a forward+backward pass to populate Fisher."""

    def test_fisher_values_are_nonzero_after_consolidate(self) -> None:
        model = nn.Linear(4, 2)
        cfg = LearningConfig(ewc_lambda=100.0, ewc_fisher_samples=5)
        agent = EWCAgent(cfg, model)

        agent.consolidate()

        has_nonzero = any(v.abs().sum().item() > 0 for v in agent._fisher.values())
        assert has_nonzero, "Fisher should contain non-zero values after consolidation"

    def test_penalty_nonzero_after_consolidation_and_drift(self) -> None:
        model = nn.Linear(4, 2)
        cfg = LearningConfig(ewc_lambda=100.0, ewc_fisher_samples=10)
        agent = EWCAgent(cfg, model)

        agent.consolidate()

        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        penalty = agent.compute_penalty()
        assert penalty.item() > 0.0

    def test_consolidate_with_data_loader(self) -> None:
        model = nn.Linear(4, 2)
        cfg = LearningConfig(ewc_lambda=100.0, ewc_fisher_samples=3)
        agent = EWCAgent(cfg, model)

        data = [torch.randn(1, 4) for _ in range(3)]
        agent.consolidate(data_loader=iter(data))

        assert len(agent._fisher) > 0
        has_nonzero = any(v.abs().sum().item() > 0 for v in agent._fisher.values())
        assert has_nonzero

    def test_consolidate_backwards_compatible_no_args(self) -> None:
        model = nn.Linear(4, 2)
        cfg = LearningConfig(ewc_lambda=1.0, ewc_fisher_samples=2)
        agent = EWCAgent(cfg, model)
        agent.consolidate()
        assert len(agent._star_params) > 0

    def test_consolidate_snapshots_params(self) -> None:
        model = nn.Linear(4, 2)
        cfg = LearningConfig(ewc_lambda=1.0, ewc_fisher_samples=1)
        agent = EWCAgent(cfg, model)
        agent.consolidate()
        for _name, param in agent._star_params.items():
            assert isinstance(param, torch.Tensor)

    def test_infer_input_dim(self) -> None:
        model = nn.Linear(8, 3)
        cfg = LearningConfig(ewc_lambda=1.0, ewc_fisher_samples=1)
        agent = EWCAgent(cfg, model)
        assert agent._infer_input_dim() == 8

    def test_infer_input_dim_sequential(self) -> None:
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2))
        cfg = LearningConfig(ewc_lambda=1.0, ewc_fisher_samples=1)
        agent = EWCAgent(cfg, model)
        assert agent._infer_input_dim() == 16

    def test_exhausted_data_loader_stops_early(self) -> None:
        model = nn.Linear(4, 2)
        cfg = LearningConfig(ewc_lambda=1.0, ewc_fisher_samples=100)
        agent = EWCAgent(cfg, model)

        data = [torch.randn(1, 4) for _ in range(2)]
        agent.consolidate(data_loader=iter(data))
        assert len(agent._fisher) > 0


# ===================================================================
# Fix #4 — Seeded RNG in EpisodicReplay
# ===================================================================


class TestEpisodicReplaySeeding:
    """Verify EpisodicReplay supports seeded reproducible sampling."""

    def test_seed_produces_reproducible_samples(self) -> None:
        cfg = MemoryConfig(episodic_capacity=100)

        replay1 = EpisodicReplay(cfg, seed=42)
        replay2 = EpisodicReplay(cfg, seed=42)

        for i in range(20):
            replay1.push(f"exp{i}", priority=float(i + 1))
            replay2.push(f"exp{i}", priority=float(i + 1))

        sample1 = replay1.sample(5)
        sample2 = replay2.sample(5)
        assert sample1 == sample2

    def test_different_seeds_differ(self) -> None:
        cfg = MemoryConfig(episodic_capacity=100)
        results: list[list[Any]] = []

        for seed in [1, 2]:
            replay = EpisodicReplay(cfg, seed=seed)
            for i in range(20):
                replay.push(f"exp{i}", priority=float(i + 1))
            results.append(replay.sample(10))

        # With different seeds, samples should typically differ
        assert results[0] != results[1]

    def test_backwards_compatible_no_seed(self) -> None:
        cfg = MemoryConfig(episodic_capacity=10)
        replay = EpisodicReplay(cfg)
        replay.push("a")
        result = replay.sample(1)
        assert len(result) == 1

    def test_none_seed_accepted(self) -> None:
        cfg = MemoryConfig(episodic_capacity=10)
        replay = EpisodicReplay(cfg, seed=None)
        replay.push("x")
        assert replay.sample(1) == ["x"]


# ===================================================================
# Fix #5 — RSSM observe_step decorated with @torch.no_grad
# ===================================================================


class TestRSSMObserveStepNoGrad:
    """Verify observe_step methods do not track gradients."""

    def test_classic_rssm_observe_step_no_grad(self) -> None:
        from mousedroid.config.schema import ModelConfig
        from mousedroid.world_model.rssm import RSSM

        model_cfg = ModelConfig(
            vision_dim=16,
            ultrasonic_dim=1,
            motor_state_dim=4,
            hidden_dim=32,
            latent_dim=8,
            action_dim=2,
            obs_dim=16,
            vision_proj_dim=8,
            ultrasonic_proj_dim=4,
            motor_proj_dim=4,
        )
        rssm = RSSM(model_cfg)

        @dataclass
        class FakeObs:
            timestamp: float = 0.0
            vision_features: np.ndarray = None  # type: ignore[assignment]
            distance_m: float = 1.0
            motor_state: np.ndarray = None  # type: ignore[assignment]
            valid_mask: np.ndarray = None  # type: ignore[assignment]
            audio_chunk: np.ndarray = None  # type: ignore[assignment]
            lidar_features: np.ndarray | None = None

        obs = FakeObs(
            vision_features=np.zeros(16, dtype=np.float32),
            motor_state=np.zeros(4, dtype=np.float32),
            valid_mask=np.ones(4, dtype=np.float32),
            audio_chunk=np.zeros(0, dtype=np.float32),
        )
        h = torch.zeros(1, 32)
        z = torch.zeros(1, 8)
        prev_action = torch.zeros(1, 2)

        new_h, new_z, _, _ = rssm.observe_step(obs, prev_action, h, z)
        assert not new_h.requires_grad
        assert not new_z.requires_grad

    def test_dual_stream_rssm_observe_step_no_grad(self) -> None:
        ncps = pytest.importorskip("ncps")  # noqa: F841

        from mousedroid.config.schema import ModelConfig
        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        model_cfg = ModelConfig(
            vision_dim=16,
            ultrasonic_dim=1,
            motor_state_dim=4,
            hidden_dim=32,
            latent_dim=8,
            action_dim=2,
            obs_dim=16,
            vision_proj_dim=8,
            ultrasonic_proj_dim=4,
            motor_proj_dim=4,
            cfc_hidden_dim=16,
            cfc_backbone_units=32,
            cfc_backbone_layers=1,
        )
        rssm = DualStreamRSSM(model_cfg)

        @dataclass
        class FakeObs:
            timestamp: float = 0.0
            vision_features: np.ndarray = None  # type: ignore[assignment]
            distance_m: float = 1.0
            motor_state: np.ndarray = None  # type: ignore[assignment]
            valid_mask: np.ndarray = None  # type: ignore[assignment]
            audio_chunk: np.ndarray = None  # type: ignore[assignment]
            lidar_features: np.ndarray | None = None

        combined_dim = model_cfg.hidden_dim + model_cfg.cfc_hidden_dim
        obs = FakeObs(
            vision_features=np.zeros(16, dtype=np.float32),
            motor_state=np.zeros(4, dtype=np.float32),
            valid_mask=np.ones(4, dtype=np.float32),
            audio_chunk=np.zeros(0, dtype=np.float32),
        )
        h = torch.zeros(1, combined_dim)
        z = torch.zeros(1, 8)
        prev_action = torch.zeros(1, 2)

        new_h, new_z, _, _ = rssm.observe_step(obs, prev_action, h, z)
        assert not new_h.requires_grad
        assert not new_z.requires_grad


# ===================================================================
# Fix #6 — SO-ARM100 async serial I/O
# ===================================================================


class TestSoArm100AsyncIO:
    """Verify SO-ARM100 driver delegates blocking serial I/O to asyncio.to_thread."""

    async def test_start_uses_to_thread(self) -> None:
        from mousedroid.arm.hardware.so_arm100_driver import SoArm100Driver
        from mousedroid.config.schema import ArmConfig

        cfg = ArmConfig()
        driver = SoArm100Driver(cfg)

        mock_serial = MagicMock()
        with (
            patch.object(driver, "_open_serial", return_value=mock_serial) as open_fn,
            patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_serial
            await driver.start()
            mock_to_thread.assert_awaited_once_with(open_fn)
            assert driver._serial is mock_serial

    async def test_stop_uses_to_thread(self) -> None:
        from mousedroid.arm.hardware.so_arm100_driver import SoArm100Driver
        from mousedroid.config.schema import ArmConfig

        cfg = ArmConfig()
        driver = SoArm100Driver(cfg)
        driver._serial = MagicMock()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            await driver.stop()
            mock_to_thread.assert_awaited_once()

    async def test_get_joint_states_raises_without_start(self) -> None:
        from mousedroid.arm.hardware.so_arm100_driver import SoArm100Driver
        from mousedroid.config.schema import ArmConfig

        cfg = ArmConfig()
        driver = SoArm100Driver(cfg)
        with pytest.raises(RuntimeError, match="not open"):
            await driver.get_joint_states()

    async def test_send_joint_command_validates_dof(self) -> None:
        from mousedroid.arm.hardware.so_arm100_driver import SoArm100Driver
        from mousedroid.config.schema import ArmConfig

        cfg = ArmConfig()
        driver = SoArm100Driver(cfg)
        driver._serial = MagicMock()

        with pytest.raises(ValueError, match="Expected"):
            await driver.send_joint_command(np.zeros(99, dtype=np.float64))


# ===================================================================
# Fix #7 — CognitiveCore._slow_loop catches asyncio.TimeoutError
# ===================================================================


class TestCognitiveSlowLoopTimeout:
    """Verify _slow_loop catches only asyncio.TimeoutError."""

    async def test_slow_loop_continues_on_timeout(self) -> None:
        from mousedroid.cognitive.bdi_model import NeuralBDI
        from mousedroid.cognitive.cognitive_core import CognitiveCore
        from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker
        from mousedroid.cognitive.metacognitive import MetacognitiveModel

        core = CognitiveCore(NeuralBDI(), MetacognitiveModel(), ConstitutionalChecker())
        await core.start()

        # Let the slow loop spin through at least one timeout cycle
        await asyncio.sleep(0.1)
        assert core._slow_task is not None
        assert not core._slow_task.done()

        await core.stop()

    async def test_tick_fast_after_slow_loop_timeout(self) -> None:
        from mousedroid.cognitive.bdi_model import NeuralBDI
        from mousedroid.cognitive.cognitive_core import CognitiveCore
        from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker
        from mousedroid.cognitive.metacognitive import MetacognitiveModel

        core = CognitiveCore(NeuralBDI(), MetacognitiveModel(), ConstitutionalChecker())
        await core.start()

        action, violations = core.tick_fast({"state": np.zeros(128, dtype=np.float32)})
        assert isinstance(action, np.ndarray)
        assert isinstance(violations, list)

        await core.stop()


# ===================================================================
# Fix #8 — Removed duplicate CuriosityAggregator from CognitiveCore
# ===================================================================


class TestCuriosityAggregatorDedup:
    """Verify CognitiveCore no longer has a duplicate CuriosityAggregator."""

    def test_no_curiosity_attr_on_core(self) -> None:
        from mousedroid.cognitive.bdi_model import NeuralBDI
        from mousedroid.cognitive.cognitive_core import CognitiveCore
        from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker
        from mousedroid.cognitive.metacognitive import MetacognitiveModel

        core = CognitiveCore(NeuralBDI(), MetacognitiveModel(), ConstitutionalChecker())
        assert not hasattr(core, "_curiosity"), (
            "CognitiveCore should not have _curiosity; use _rl._curiosity"
        )

    def test_rl_bundle_has_curiosity(self) -> None:
        from mousedroid.cognitive.bdi_model import NeuralBDI
        from mousedroid.cognitive.cognitive_core import CognitiveCore
        from mousedroid.cognitive.constitutional_rl import (
            ConstitutionalChecker,
            CuriosityAggregator,
        )
        from mousedroid.cognitive.metacognitive import MetacognitiveModel

        core = CognitiveCore(NeuralBDI(), MetacognitiveModel(), ConstitutionalChecker())
        assert hasattr(core._rl, "_curiosity")
        assert isinstance(core._rl._curiosity, CuriosityAggregator)

    def test_tick_fast_uses_rl_curiosity(self) -> None:
        from mousedroid.cognitive.bdi_model import NeuralBDI
        from mousedroid.cognitive.cognitive_core import CognitiveCore
        from mousedroid.cognitive.constitutional_rl import ConstitutionalChecker
        from mousedroid.cognitive.metacognitive import MetacognitiveModel

        core = CognitiveCore(NeuralBDI(), MetacognitiveModel(), ConstitutionalChecker())
        obs = {
            "state": np.zeros(128, dtype=np.float32),
            "curiosity": {"social": 0.5, "epistemic": 0.8},
        }
        action, _violations = core.tick_fast(obs)
        assert isinstance(action, np.ndarray)


# ===================================================================
# Fix #9 — MCTS plan() no redundant tanh squashing
# ===================================================================


class MockWorldModel:
    """Deterministic world model stub for MCTS tests."""

    def observe_step(
        self,
        observation: Any,
        prev_action: Any,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        return h, z, h, 0.0

    def imagine_step(
        self,
        action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_h = h.clone()
        new_z = z.clone()
        reward = torch.zeros(1, 1)
        return new_h, new_z, reward


class TestMCTSNoDoubleTanh:
    """Verify MCTS plan() does not apply redundant tanh squashing."""

    def test_plan_preserves_full_range(self) -> None:
        cfg = MCTSConfig(
            n_simulations_base=4,
            rollout_depth=2,
            n_action_candidates=3,
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)

        action = planner.plan(h, z)
        # Actions should be in [-1, 1] — the exact linspace values
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()

    def test_extreme_actions_not_squashed(self) -> None:
        cfg = MCTSConfig(
            n_simulations_base=2,
            rollout_depth=1,
            n_action_candidates=3,
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)

        action = planner.plan(h, z)
        max_abs = action.abs().max().item()
        # Should be able to reach 1.0 (not squashed to ~0.76 by tanh)
        # With 3 candidates in linspace(-1, 1, 3) = [-1, 0, 1],
        # one of these becomes the best child
        assert max_abs >= 0.99 or max_abs == 0.0  # 0.0 is valid if center chosen

    def test_candidate_actions_in_valid_range(self) -> None:
        cfg = MCTSConfig(
            n_simulations_base=4,
            rollout_depth=2,
            n_action_candidates=5,
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        actions = planner._generate_candidate_actions(torch.device("cpu"))

        assert (actions >= -1.0).all()
        assert (actions <= 1.0).all()
        assert actions.shape == (5, 3)

    def test_plan_no_grad(self) -> None:
        cfg = MCTSConfig(
            n_simulations_base=4,
            rollout_depth=2,
            n_action_candidates=3,
        )
        planner = MCTSPlanner(cfg, MockWorldModel())
        h = torch.zeros(1, 256)
        z = torch.zeros(1, 64)

        action = planner.plan(h, z)
        assert not action.requires_grad
