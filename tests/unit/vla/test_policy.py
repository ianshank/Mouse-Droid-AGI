"""Unit tests for the VLA policy package (Phase 3a).

Covers:
* ``VLAPolicyProtocol`` runtime conformance for ``MockVLA``.
* Action shape, dtype, no-grad path, and determinism.
* ``MockVLA`` constructor validation.
* Factory ``build_vla_policy`` gating + canned-action plumbing.
"""

from __future__ import annotations

import pytest
import torch

from mousedroid.config.schema import Settings, VLAConfig
from mousedroid.factory import build_vla_policy
from mousedroid.vla import MockVLA, VLAAction, VLAObservation, VLAPolicyProtocol


def _make_cfg(**vla: object) -> Settings:
    """Build a fresh ``Settings`` with a custom ``vla`` block."""
    cfg = Settings(mock_hardware=True)
    cfg.vla = VLAConfig(**vla)  # type: ignore[arg-type]
    return cfg


class TestVLAObservation:
    def test_immutable(self) -> None:
        obs = VLAObservation(h=torch.zeros(2), z=torch.zeros(3))
        with pytest.raises(AttributeError):  # frozen dataclass
            obs.instruction = "x"  # type: ignore[misc]

    def test_default_instruction_is_empty(self) -> None:
        obs = VLAObservation(h=torch.zeros(1), z=torch.zeros(1))
        assert obs.instruction == ""

    def test_carries_tensors(self) -> None:
        h = torch.tensor([1.0, 2.0])
        z = torch.tensor([3.0])
        obs = VLAObservation(h=h, z=z, instruction="go")
        assert torch.equal(obs.h, h)
        assert torch.equal(obs.z, z)
        assert obs.instruction == "go"


class TestVLAActionShape:
    def test_default_confidence(self) -> None:
        act = VLAAction(action=torch.zeros(3))
        assert act.confidence == 1.0


class TestMockVLAValidation:
    def test_rejects_zero_action_dim(self) -> None:
        with pytest.raises(ValueError, match="action_dim"):
            MockVLA(action_dim=0)

    def test_rejects_negative_action_dim(self) -> None:
        with pytest.raises(ValueError, match="action_dim"):
            MockVLA(action_dim=-1)

    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            MockVLA(action_dim=3, confidence=1.5)

    def test_rejects_confidence_below_zero(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            MockVLA(action_dim=3, confidence=-0.1)

    def test_rejects_canned_action_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="canned_action shape"):
            MockVLA(action_dim=3, canned_action=torch.zeros(4))


class TestMockVLAProtocol:
    def test_satisfies_protocol(self) -> None:
        policy = MockVLA(action_dim=3)
        assert isinstance(policy, VLAPolicyProtocol)

    def test_name_default(self) -> None:
        policy = MockVLA(action_dim=3)
        assert policy.name == "mock_vla"

    def test_name_override(self) -> None:
        policy = MockVLA(action_dim=3, name="custom")
        assert policy.name == "custom"


class TestMockVLAPredict:
    def _obs(self) -> VLAObservation:
        return VLAObservation(h=torch.zeros(4), z=torch.zeros(2))

    def test_default_returns_zero_action(self) -> None:
        policy = MockVLA(action_dim=3)
        result = policy.predict(self._obs())
        assert torch.equal(result.action, torch.zeros(3))

    def test_action_shape_matches_action_dim(self) -> None:
        policy = MockVLA(action_dim=5)
        result = policy.predict(self._obs())
        assert result.action.shape == (5,)

    def test_action_dtype_is_float32(self) -> None:
        policy = MockVLA(action_dim=3)
        result = policy.predict(self._obs())
        assert result.action.dtype == torch.float32

    def test_canned_action_returned(self) -> None:
        canned = torch.tensor([0.5, -0.25, 0.1])
        policy = MockVLA(action_dim=3, canned_action=canned)
        result = policy.predict(self._obs())
        assert torch.allclose(result.action, canned)

    def test_returns_clone_not_reference(self) -> None:
        canned = torch.tensor([1.0, 2.0, 3.0])
        policy = MockVLA(action_dim=3, canned_action=canned)
        a = policy.predict(self._obs()).action
        a[0] = 99.0
        b = policy.predict(self._obs()).action
        assert b[0].item() == 1.0  # not corrupted by mutation of prior return

    def test_deterministic_across_calls(self) -> None:
        policy = MockVLA(action_dim=3, canned_action=torch.tensor([0.1, 0.2, 0.3]))
        first = policy.predict(self._obs()).action
        second = policy.predict(self._obs()).action
        assert torch.equal(first, second)

    def test_confidence_propagated(self) -> None:
        policy = MockVLA(action_dim=3, confidence=0.42)
        assert policy.predict(self._obs()).confidence == 0.42

    def test_inference_runs_under_no_grad(self) -> None:
        policy = MockVLA(action_dim=3)
        result = policy.predict(self._obs())
        assert result.action.requires_grad is False


class TestBuildVLAPolicy:
    def test_disabled_by_default(self) -> None:
        cfg = Settings(mock_hardware=True)
        assert build_vla_policy(cfg) is None

    def test_explicit_none_backend(self) -> None:
        cfg = _make_cfg(backend="none")
        assert build_vla_policy(cfg) is None

    def test_mock_backend_returns_mockvla(self) -> None:
        cfg = _make_cfg(backend="mock")
        policy = build_vla_policy(cfg)
        assert isinstance(policy, MockVLA)
        assert isinstance(policy, VLAPolicyProtocol)

    def test_mock_backend_uses_action_dim_from_model(self) -> None:
        cfg = _make_cfg(backend="mock")
        policy = build_vla_policy(cfg)
        assert policy is not None
        obs = VLAObservation(h=torch.zeros(1), z=torch.zeros(1))
        assert policy.predict(obs).action.shape == (cfg.model.action_dim,)

    def test_mock_backend_with_canned_action(self) -> None:
        cfg = _make_cfg(
            backend="mock",
            canned_action=[0.0] * Settings(mock_hardware=True).model.action_dim,
        )
        policy = build_vla_policy(cfg)
        assert policy is not None

    def test_canned_action_length_mismatch_raises(self) -> None:
        cfg = _make_cfg(backend="mock", canned_action=[0.0])  # too short
        with pytest.raises(ValueError, match="canned_action length"):
            build_vla_policy(cfg)

    def test_distilled_onnx_reserved(self) -> None:
        cfg = _make_cfg(backend="distilled_onnx")
        with pytest.raises(NotImplementedError, match="Phase 3b"):
            build_vla_policy(cfg)

    def test_confidence_propagated_through_factory(self) -> None:
        cfg = _make_cfg(backend="mock", confidence=0.7)
        policy = build_vla_policy(cfg)
        assert policy is not None
        obs = VLAObservation(h=torch.zeros(1), z=torch.zeros(1))
        assert policy.predict(obs).confidence == 0.7
