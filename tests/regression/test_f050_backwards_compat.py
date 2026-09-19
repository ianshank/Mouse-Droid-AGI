"""Backwards compatibility — F-050 must be additive only.

The observe-step seam adds a keyword-only, defaulted ``metrics`` parameter and
renames each engine's inference body to ``_observe_step_impl`` behind an
unchanged public ``observe_step``. Nothing about that may change behaviour for a
caller that does not opt in.

Pinned here:

* Every pre-existing construction and call shape still works untouched.
* No new YAML key: every shipped ``config/*.yaml`` loads unchanged, and none
  gains a ``world_model:`` block (design D-7 — ``check_config_compat.py``
  validates changed config files against a pinned schema where
  ``WorldModelConfig`` is ``extra="ignore"``, so such a key would pass the gate
  and then be silently dropped).
* ``WorldModelProtocol`` conformance and the ``(Tensor, Tensor, Tensor, float)``
  return contract are unchanged.
* The sink never enters ``state_dict``, so checkpoints round-trip identically.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from numpy.typing import NDArray

from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.factory.world_model import build_world_model
from mousedroid.world_model.protocol import WorldModelProtocol
from mousedroid.world_model.rssm import RSSM

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"


@dataclass
class _Obs:
    """Minimal ``ObservationProtocol`` stub."""

    timestamp: float = 0.0
    distance_m: float = 1.5
    n_modalities: int = 4
    vision_features: NDArray[np.float32] = field(default_factory=lambda: np.zeros(16, np.float32))
    motor_state: NDArray[np.float32] = field(default_factory=lambda: np.zeros(4, np.float32))
    audio_chunk: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    valid_mask: NDArray[np.float32] = field(default_factory=lambda: np.ones(4, np.float32))
    lidar_features: NDArray[np.float32] | None = None


def _cfg(*, cfc_hidden_dim: int = 0) -> ModelConfig:
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
        cfc_hidden_dim=cfc_hidden_dim,
        cfc_backbone_units=32,
        cfc_backbone_layers=1,
    )


def _yaml_files() -> list[Path]:
    """Every shipped top-level config overlay (excludes nested tool configs)."""
    return sorted(p for p in _CONFIG_DIR.glob("*.yaml") if p.is_file())


class TestPreExistingCallShapes:
    """Old call sites must keep working verbatim."""

    def test_rssm_constructs_positionally_without_metrics(self) -> None:
        assert RSSM(_cfg()) is not None

    def test_dual_stream_constructs_without_metrics(self) -> None:
        pytest.importorskip("ncps")
        from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

        assert DualStreamRSSM(_cfg(cfc_hidden_dim=16)) is not None

    def test_build_world_model_single_argument_call(self) -> None:
        assert build_world_model(Settings(mock_hardware=True, model=_cfg())) is not None

    def test_observe_step_returns_the_documented_tuple(self) -> None:
        cfg = _cfg()
        model = RSSM(cfg)
        new_h, new_z, obs_embed, surprise = model.observe_step(
            _Obs(),
            torch.zeros(1, cfg.action_dim),
            torch.zeros(1, cfg.hidden_dim),
            torch.zeros(1, cfg.latent_dim),
        )
        assert isinstance(new_h, torch.Tensor)
        assert isinstance(new_z, torch.Tensor)
        assert isinstance(obs_embed, torch.Tensor)
        assert isinstance(surprise, float), "surprise must stay a Python float"

    def test_protocol_conformance_is_unchanged(self) -> None:
        assert isinstance(RSSM(_cfg()), WorldModelProtocol)


class TestCheckpointCompatibility:
    """The sink must not leak into persisted state."""

    def test_metrics_sink_is_absent_from_state_dict(self) -> None:
        from mousedroid.config.schema import Settings as _Settings
        from mousedroid.telemetry.metrics.registry import MetricsRegistry

        registry = MetricsRegistry(_Settings(mock_hardware=True).metrics)
        keys_without = set(RSSM(_cfg()).state_dict().keys())
        keys_with = set(RSSM(_cfg(), metrics=registry).state_dict().keys())
        assert keys_with == keys_without
        assert not any("metric" in key.lower() for key in keys_with)

    def test_checkpoint_round_trips_between_wired_and_unwired_engines(self) -> None:
        from mousedroid.config.schema import Settings as _Settings
        from mousedroid.telemetry.metrics.registry import MetricsRegistry

        registry = MetricsRegistry(_Settings(mock_hardware=True).metrics)
        source = RSSM(_cfg(), metrics=registry)
        target = RSSM(_cfg())
        target.load_state_dict(source.state_dict())


class TestRefinementCandidateCopying:
    """``copy.deepcopy`` of a live engine must keep working.

    ``learning/on_device/rssm_refiner.py`` deepcopies ``self._base_rssm`` to build
    a refinement candidate. Wiring a :class:`MetricsRegistry` onto the module put
    a ``threading.Lock`` inside the copied state, which broke that copy with
    ``TypeError: cannot pickle '_thread.lock' object`` — caught by
    ``tests/integration/test_on_device_sim_soak.py`` before it shipped.
    ``ObserveStepTimingMixin`` drops the sink on copy, which is also the correct
    behaviour: an off-loop candidate must not write to the production histogram.
    """

    @staticmethod
    def _registry() -> object:
        from mousedroid.config.schema import Settings as _Settings
        from mousedroid.telemetry.metrics.registry import MetricsRegistry

        return MetricsRegistry(_Settings(mock_hardware=True).metrics)

    def test_deepcopy_of_a_wired_engine_succeeds(self) -> None:
        assert copy.deepcopy(RSSM(_cfg(), metrics=self._registry())) is not None  # type: ignore[arg-type]

    def test_the_copied_engine_is_untimed(self) -> None:
        candidate = copy.deepcopy(RSSM(_cfg(), metrics=self._registry()))  # type: ignore[arg-type]
        assert candidate._metrics is None  # type: ignore[attr-defined]

    def test_the_source_engine_keeps_reporting(self) -> None:
        registry = self._registry()
        model = RSSM(_cfg(), metrics=registry)  # type: ignore[arg-type]
        copy.deepcopy(model)
        assert model._metrics is registry  # type: ignore[attr-defined]

    def test_the_copied_engine_still_runs_inference(self) -> None:
        cfg = _cfg()
        candidate = copy.deepcopy(RSSM(cfg, metrics=self._registry()))  # type: ignore[arg-type]
        _, _, _, surprise = candidate.observe_step(
            _Obs(),
            torch.zeros(1, cfg.action_dim),
            torch.zeros(1, cfg.hidden_dim),
            torch.zeros(1, cfg.latent_dim),
        )
        assert isinstance(surprise, float)

    def test_copied_weights_match_the_source(self) -> None:
        source = RSSM(_cfg(), metrics=self._registry())  # type: ignore[arg-type]
        candidate = copy.deepcopy(source)
        for key, value in source.state_dict().items():
            assert torch.equal(candidate.state_dict()[key], value), key


class TestConfigUnchanged:
    """No new YAML key, and every overlay still parses."""

    def test_no_shipped_overlay_declares_a_world_model_block(self) -> None:
        offenders = [
            path.name
            for path in _yaml_files()
            if isinstance(loaded := yaml.safe_load(path.read_text(encoding="utf-8")), dict)
            and "world_model" in loaded
        ]
        assert offenders == [], (
            "design D-7: a tracked world_model: key passes config-compat against the "
            f"pinned schema and is then silently dropped — offenders: {offenders}"
        )

    def test_every_shipped_overlay_still_parses(self) -> None:
        for path in _yaml_files():
            assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path.name

    def test_world_model_defaults_are_unchanged(self) -> None:
        """The engine default must stay ``torch`` so existing rovers do not move."""
        settings = Settings(mock_hardware=True)
        assert settings.world_model.engine == "torch"
        assert settings.world_model.onnx_warmup_iterations == 1

    def test_default_settings_build_an_unwired_engine(self) -> None:
        """Absent an explicit registry, behaviour is byte-identical to pre-F-050."""
        model = build_world_model(Settings(mock_hardware=True, model=_cfg()))
        assert model._metrics is None  # type: ignore[attr-defined]
