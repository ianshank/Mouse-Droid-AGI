"""Unit tests for ``DualStreamRSSM.observe_step_traceable``.

The traceable variant mirrors :meth:`DualStreamRSSM.observe_step` but
returns ``surprise`` as a scalar ``Tensor`` (not a Python ``float``) and
accepts unpacked tensors directly (not an ``ObservationProtocol``). This
makes it consumable by ``torch.onnx.export`` and by the planned
``DualStreamRSSMOnnx`` runtime class — neither of which can trace
``ObservationProtocol`` Python objects.

The public ``observe_step`` continues to expose its existing contract;
this is purely an additive internal helper that the production path
also uses (to avoid drift between PyTorch and ONNX outputs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

pytest.importorskip("ncps")

from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
from mousedroid.world_model.observation_packer import pack_observation


@dataclass
class _StubObservation:
    """Minimal ObservationProtocol-compatible stub for tests."""

    timestamp: float = 0.0
    vision_features: NDArray[np.float32] | None = None
    distance_m: float = 1.5
    motor_state: NDArray[np.float32] | None = None
    audio_chunk: NDArray[np.float32] | None = None
    valid_mask: NDArray[np.float32] | None = None
    n_modalities: int = 4
    lidar_features: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if self.vision_features is None:
            self.vision_features = np.zeros(16, dtype=np.float32)
        if self.motor_state is None:
            self.motor_state = np.zeros(4, dtype=np.float32)
        if self.audio_chunk is None:
            self.audio_chunk = np.zeros(0, dtype=np.float32)
        if self.valid_mask is None:
            self.valid_mask = np.ones(4, dtype=np.float32)


def _make_cfg(cfc_dim: int = 16, hidden_dim: int = 32, latent_dim: int = 8) -> ModelConfig:
    return ModelConfig(
        vision_dim=16,
        ultrasonic_dim=1,
        ultrasonic_proj_dim=4,
        motor_state_dim=4,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        action_dim=2,
        obs_dim=16,
        vision_proj_dim=8,
        motor_proj_dim=4,
        cfc_hidden_dim=cfc_dim,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )


@pytest.fixture
def model_and_inputs() -> tuple[DualStreamRSSM, dict[str, torch.Tensor]]:
    """Return a constructed model + the unpacked tensor dict + recurrent state."""
    cfg = _make_cfg()
    model = DualStreamRSSM(cfg)
    model.train(False)  # inference mode

    obs = _StubObservation()
    packed = pack_observation(obs, cfg, device=torch.device("cpu"))
    prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
    h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
    z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)

    inputs = {
        "vision": packed.vision,
        "motor": packed.motor,
        "valid_mask": packed.valid_mask,
        "ultrasonic": packed.ultrasonic,
        "audio": packed.audio,
        "lidar": packed.lidar,
        "prev_action": prev_action,
        "h": h,
        "z": z,
    }
    return model, inputs


class TestObserveStepTraceable:
    """Cross-checks the traceable variant against the public observe_step."""

    def test_returns_4_tuple_with_tensor_surprise(
        self,
        model_and_inputs: tuple[DualStreamRSSM, dict[str, torch.Tensor]],
    ) -> None:
        """observe_step_traceable returns (new_h, new_z, obs_embed, surprise_tensor)."""
        model, inputs = model_and_inputs
        result = model.observe_step_traceable(**inputs)
        assert len(result) == 4
        new_h, new_z, obs_embed, surprise = result
        assert isinstance(new_h, torch.Tensor)
        assert isinstance(new_z, torch.Tensor)
        assert isinstance(obs_embed, torch.Tensor)
        # surprise is a Tensor (scalar shape), NOT a Python float.
        assert isinstance(surprise, torch.Tensor)

    def test_shapes_match_observe_step(
        self,
        model_and_inputs: tuple[DualStreamRSSM, dict[str, torch.Tensor]],
    ) -> None:
        """Traceable output shapes mirror observe_step output shapes."""
        model, inputs = model_and_inputs
        new_h, new_z, obs_embed, surprise = model.observe_step_traceable(**inputs)
        # Combined hidden state shape: (1, hidden_dim + cfc_hidden_dim) = (1, 32 + 16) = (1, 48)
        assert new_h.shape == (1, 32 + 16)
        assert new_z.shape == (1, 8)
        assert obs_embed.shape == (1, 16)
        # surprise is a scalar — shape () or (1,) is acceptable
        assert surprise.numel() == 1

    def test_surprise_tensor_matches_float_surprise(
        self,
        model_and_inputs: tuple[DualStreamRSSM, dict[str, torch.Tensor]],
    ) -> None:
        """Traceable ``surprise.item()`` must equal observe_step's float surprise.

        This is the integration guarantee: refactoring observe_step to use
        the traceable variant under the hood must not change the public
        contract (a Python ``float``).
        """
        model, inputs = model_and_inputs

        # Run traceable
        torch.manual_seed(0)
        _, _, _, surprise_tensor = model.observe_step_traceable(**inputs)
        traceable_value = float(surprise_tensor.item())

        # Run observe_step on the same model + inputs
        obs = _StubObservation()
        torch.manual_seed(0)
        _, _, _, float_value = model.observe_step(
            obs, inputs["prev_action"], inputs["h"], inputs["z"]
        )
        assert traceable_value == pytest.approx(float_value, abs=1e-5)


class TestObserveStepBackwardsCompat:
    """The public ``observe_step`` still returns ``float`` surprise — no API change."""

    def test_observe_step_returns_float_surprise(self) -> None:
        cfg = _make_cfg()
        model = DualStreamRSSM(cfg)
        model.train(False)

        obs = _StubObservation()
        prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)

        new_h, new_z, obs_embed, surprise = model.observe_step(obs, prev_action, h, z)
        assert isinstance(surprise, float)
        assert isinstance(new_h, torch.Tensor)
        assert isinstance(new_z, torch.Tensor)
        assert isinstance(obs_embed, torch.Tensor)
