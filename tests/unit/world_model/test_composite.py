"""Tests for ``CompositeWorldModel`` — observe / imagine engine split.

The composite resolves a critical bug in ``cfg.world_model.engine =
"onnx_trt"``: the ONNX export ships only ``observe_step``, but
``MCTSPlanner.plan()`` calls ``imagine_step`` during rollouts. Without
the composite, flipping the engine would crash the planner at runtime.

These tests pin:
1. ``observe_step`` delegates to the observe engine
2. ``imagine_step`` delegates to the imagine engine
3. ``get_safety_trace`` delegates only when the imagine engine
   implements ``SafetyTraceProtocol`` (raises a clear error otherwise)
4. The composite conforms to ``WorldModelProtocol`` (so the factory's
   return type stays consistent)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

pytest.importorskip("ncps")


from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.composite import CompositeWorldModel
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
from mousedroid.world_model.protocol import WorldModelProtocol


@dataclass
class _StubObservation:
    timestamp: float = 0.0
    vision_features: NDArray[np.float32] | None = None
    distance_m: float = 1.5
    motor_state: NDArray[np.float32] | None = None
    audio_chunk: NDArray[np.float32] | None = None
    valid_mask: NDArray[np.float32] | None = None
    n_modalities: int = 5
    lidar_features: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if self.vision_features is None:
            self.vision_features = np.zeros(16, dtype=np.float32)
        if self.motor_state is None:
            self.motor_state = np.zeros(4, dtype=np.float32)
        if self.audio_chunk is None:
            self.audio_chunk = np.zeros(0, dtype=np.float32)
        if self.valid_mask is None:
            self.valid_mask = np.ones(5, dtype=np.float32)


def _cfg() -> ModelConfig:
    return ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        ultrasonic_proj_dim=4,
        motor_state_dim=4,
        hidden_dim=32,
        latent_dim=8,
        action_dim=2,
        obs_dim=16,
        vision_proj_dim=8,
        motor_proj_dim=4,
        cfc_hidden_dim=16,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )


class _RecordingObserveEngine:
    """Records observe_step calls; imagine_step delegates raise to surface routing bugs."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.observe_calls: list[tuple[Any, ...]] = []
        self._cfg = cfg

    def observe_step(
        self,
        observation: Any,
        prev_action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        self.observe_calls.append((observation, prev_action, h, z))
        # Return shapes that match the protocol contract.
        new_h = h.clone()
        new_z = z.clone()
        obs_embed = torch.zeros(1, self._cfg.obs_dim, dtype=torch.float32)
        return new_h, new_z, obs_embed, 0.0

    def imagine_step(
        self,
        action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Must never be called on the observe engine — surface routing bugs.
        msg = "imagine_step should never reach the observe engine in the composite"
        raise AssertionError(msg)


class _RecordingImagineEngine:
    """Records imagine_step calls; observe_step raises to surface routing bugs."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.imagine_calls: list[tuple[Any, ...]] = []
        self._cfg = cfg

    def observe_step(
        self,
        observation: Any,
        prev_action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        msg = "observe_step should never reach the imagine engine in the composite"
        raise AssertionError(msg)

    def imagine_step(
        self,
        action: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.imagine_calls.append((action, h, z))
        new_h = h.clone()
        new_z = z.clone()
        predicted_reward = torch.zeros(1, 1, dtype=torch.float32)
        return new_h, new_z, predicted_reward


class TestRouting:
    """observe_step → observe engine; imagine_step → imagine engine."""

    def test_observe_step_delegates_to_observe_engine(self) -> None:
        cfg = _cfg()
        observe = _RecordingObserveEngine(cfg)
        imagine = _RecordingImagineEngine(cfg)
        composite = CompositeWorldModel(observe_engine=observe, imagine_engine=imagine)

        obs = _StubObservation()
        prev_action = torch.zeros(1, cfg.action_dim)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim)
        z = torch.zeros(1, cfg.latent_dim)
        composite.observe_step(obs, prev_action, h, z)

        assert len(observe.observe_calls) == 1
        assert len(imagine.imagine_calls) == 0

    def test_imagine_step_delegates_to_imagine_engine(self) -> None:
        cfg = _cfg()
        observe = _RecordingObserveEngine(cfg)
        imagine = _RecordingImagineEngine(cfg)
        composite = CompositeWorldModel(observe_engine=observe, imagine_engine=imagine)

        action = torch.zeros(1, cfg.action_dim)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim)
        z = torch.zeros(1, cfg.latent_dim)
        composite.imagine_step(action, h, z)

        assert len(imagine.imagine_calls) == 1
        assert len(observe.observe_calls) == 0


class TestProtocolConformance:
    """The composite conforms to ``WorldModelProtocol`` — drop-in for the factory."""

    def test_composite_is_world_model_protocol(self) -> None:
        cfg = _cfg()
        composite = CompositeWorldModel(
            observe_engine=_RecordingObserveEngine(cfg),
            imagine_engine=_RecordingImagineEngine(cfg),
        )
        assert isinstance(composite, WorldModelProtocol)


class TestSafetyTraceDelegation:
    """``get_safety_trace`` forwards to imagine engine when supported."""

    def test_safety_trace_forwards_to_pytorch_imagine_engine(self) -> None:
        cfg = _cfg()
        # DualStreamRSSM implements SafetyTraceProtocol.
        torch_model = DualStreamRSSM(cfg)
        torch_model.train(False)
        observe = _RecordingObserveEngine(cfg)
        composite = CompositeWorldModel(observe_engine=observe, imagine_engine=torch_model)

        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim)
        trace = composite.get_safety_trace(h)
        assert trace.shape == (1, cfg.cfc_hidden_dim)

    def test_safety_trace_raises_when_imagine_engine_lacks_protocol(self) -> None:
        cfg = _cfg()
        # _RecordingImagineEngine does NOT implement SafetyTraceProtocol.
        composite = CompositeWorldModel(
            observe_engine=_RecordingObserveEngine(cfg),
            imagine_engine=_RecordingImagineEngine(cfg),
        )
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim)
        with pytest.raises(AttributeError, match="SafetyTraceProtocol"):
            composite.get_safety_trace(h)


class TestEnginePropertyAccess:
    """Properties expose the underlying engines for introspection / tests."""

    def test_engine_properties_return_supplied_engines(self) -> None:
        cfg = _cfg()
        observe = _RecordingObserveEngine(cfg)
        imagine = _RecordingImagineEngine(cfg)
        composite = CompositeWorldModel(observe_engine=observe, imagine_engine=imagine)
        assert composite.observe_engine is observe
        assert composite.imagine_engine is imagine

    def test_name_property_returns_constructor_value(self) -> None:
        cfg = _cfg()
        composite = CompositeWorldModel(
            observe_engine=_RecordingObserveEngine(cfg),
            imagine_engine=_RecordingImagineEngine(cfg),
            name="custom_label",
        )
        assert composite.name == "custom_label"


class TestMCTSPlannerIntegration:
    """Regression test for the original bug: MCTSPlanner calls imagine_step.

    Before the composite, ``engine="onnx_trt"`` returned ``DualStreamRSSMOnnx``
    whose ``imagine_step`` raised NotImplementedError → planner crashed.
    Now the composite routes imagine_step to the PyTorch engine and the
    planner sees a fully-functional WorldModelProtocol.
    """

    def test_imagine_step_does_not_raise_not_implemented(self) -> None:
        cfg = _cfg()
        observe = _RecordingObserveEngine(cfg)
        imagine = DualStreamRSSM(cfg)
        imagine.train(False)
        composite = CompositeWorldModel(observe_engine=observe, imagine_engine=imagine)

        action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)

        # No exception — full imagine_step on the PyTorch engine.
        new_h, new_z, reward = composite.imagine_step(action, h, z)
        assert new_h.shape == (1, cfg.hidden_dim + cfg.cfc_hidden_dim)
        assert new_z.shape == (1, cfg.latent_dim)
        assert reward.shape == (1, 1)
