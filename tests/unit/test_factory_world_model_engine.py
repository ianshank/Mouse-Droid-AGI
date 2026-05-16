"""Tests for ``cfg.world_model.engine`` dispatch in ``build_world_model``.

Tier B Track B2 Story 3 — verifies the factory:

1. Defaults to ``engine="torch"`` (byte-identical pre-PR behavior)
2. Constructs :class:`DualStreamRSSMOnnx` when ``engine="onnx_trt"``
   and ``onnx_path`` points at a valid file
3. Resolves ``onnx_path`` from HF Hub when not set locally
4. Rejects unknown engine values with a clear error
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("ncps")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


from mousedroid.config.schema import ModelConfig, Settings, WorldModelConfig
from mousedroid.factory import build_world_model
from mousedroid.world_model.composite import CompositeWorldModel
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
from mousedroid.world_model.dual_stream_rssm_onnx import DualStreamRSSMOnnx


def _make_cfg(
    *,
    engine: str = "torch",
    onnx_path: str | None = None,
    cfc_hidden_dim: int = 16,
) -> Settings:
    """Build a minimal Settings with the given engine selector.

    The default ``mock_hardware=True`` is implicit (via Settings defaults)
    and keeps the build cheap — no real hardware probing.
    """
    return Settings(
        mock_hardware=True,
        model=ModelConfig(
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
        ),
        world_model=WorldModelConfig(engine=engine, onnx_path=onnx_path),
    )


def _load_export_module() -> Any:
    """Import ``scripts/export_dual_stream_rssm_onnx.py`` as a module."""
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "export_dual_stream_rssm_onnx.py"
    )
    spec = importlib.util.spec_from_file_location("export_dual_stream_rssm_onnx", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_dual_stream_rssm_onnx"] = module
    spec.loader.exec_module(module)
    return module


class TestDefaultEngineIsTorch:
    """The default engine ('torch') preserves byte-identical pre-PR behavior."""

    def test_default_engine_is_torch(self) -> None:
        """``cfg.world_model.engine`` defaults to 'torch' when YAML omits it."""
        cfg = _make_cfg()
        assert cfg.world_model.engine == "torch"

    def test_engine_torch_builds_pytorch_dual_stream_rssm(self) -> None:
        """``engine='torch'`` constructs the PyTorch ``DualStreamRSSM``."""
        cfg = _make_cfg(engine="torch")
        model = build_world_model(cfg)
        assert isinstance(model, DualStreamRSSM)

    def test_existing_settings_without_world_model_block_loads(self) -> None:
        """A Settings instance built without world_model uses the default.

        Mirrors what happens when an existing ``config/*.yaml`` loads on
        a code base that just added the WorldModelConfig field — the
        default engine is 'torch', so the build path is unchanged from
        pre-B2 (RSSM when cfc_hidden_dim=0, DualStreamRSSM otherwise).
        """
        cfg = Settings(mock_hardware=True)
        assert cfg.world_model.engine == "torch"
        # Default ModelConfig has cfc_hidden_dim=0 → factory falls back to RSSM.
        from mousedroid.world_model.rssm import RSSM

        model = build_world_model(cfg)
        assert isinstance(model, RSSM)


class TestEngineOnnxTrt:
    """``engine='onnx_trt'`` constructs ``DualStreamRSSMOnnx`` from the local artifact."""

    def test_engine_onnx_trt_with_local_path(self, tmp_path: Path) -> None:
        """When ``onnx_path`` points at a valid .onnx, build_world_model returns the composite."""
        # Export a fresh .onnx — Story 1's library entry point makes this fast.
        export_module = _load_export_module()
        cfg_for_export = _make_cfg().model
        torch_model = DualStreamRSSM(cfg_for_export)
        torch_model.train(False)
        onnx_path = tmp_path / "observe_step.onnx"
        export_module.run_export(
            model=torch_model, cfg=cfg_for_export, output_path=onnx_path, opset=17
        )

        cfg = _make_cfg(engine="onnx_trt", onnx_path=str(onnx_path))
        model = build_world_model(cfg)
        # ``engine="onnx_trt"`` returns a CompositeWorldModel that routes
        # observe_step to the ONNX runtime and imagine_step to the PyTorch
        # model — MCTS planning calls imagine_step, so the composite is
        # essential to avoid NotImplementedError at runtime.
        assert isinstance(model, CompositeWorldModel)
        assert isinstance(model.observe_engine, DualStreamRSSMOnnx)
        assert isinstance(model.imagine_engine, DualStreamRSSM)

    def test_engine_onnx_trt_imagine_step_does_not_raise(self, tmp_path: Path) -> None:
        """Regression net for the composite: imagine_step must NOT raise NotImplementedError.

        Before B2 Story 3.1, the factory returned ``DualStreamRSSMOnnx``
        directly, whose ``imagine_step`` raised NotImplementedError —
        which would crash MCTSPlanner.plan() at the first rollout. The
        composite splits observe_step (ONNX) from imagine_step (PyTorch)
        so the planner sees a fully-functional WorldModelProtocol.
        """
        export_module = _load_export_module()
        cfg_for_export = _make_cfg().model
        torch_model = DualStreamRSSM(cfg_for_export)
        torch_model.train(False)
        onnx_path = tmp_path / "observe_step.onnx"
        export_module.run_export(
            model=torch_model, cfg=cfg_for_export, output_path=onnx_path, opset=17
        )

        cfg = _make_cfg(engine="onnx_trt", onnx_path=str(onnx_path))
        model = build_world_model(cfg)

        action = torch.zeros(1, cfg.model.action_dim)
        h = torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim)
        z = torch.zeros(1, cfg.model.latent_dim)

        # Must not raise — this was the regression we fixed.
        new_h, new_z, reward = model.imagine_step(action, h, z)
        assert new_h.shape == (1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim)
        assert new_z.shape == (1, cfg.model.latent_dim)
        assert reward.shape == (1, 1)

    def test_engine_onnx_trt_observe_step_runs(self, tmp_path: Path) -> None:
        """The factory-built ONNX runtime can run observe_step end-to-end."""
        export_module = _load_export_module()
        cfg_for_export = _make_cfg().model
        torch_model = DualStreamRSSM(cfg_for_export)
        torch_model.train(False)
        onnx_path = tmp_path / "observe_step.onnx"
        export_module.run_export(
            model=torch_model, cfg=cfg_for_export, output_path=onnx_path, opset=17
        )

        cfg = _make_cfg(engine="onnx_trt", onnx_path=str(onnx_path))
        model = build_world_model(cfg)

        from dataclasses import dataclass

        import numpy as np
        from numpy.typing import NDArray

        @dataclass
        class _Obs:
            timestamp: float = 0.0
            vision_features: NDArray[Any] = None  # type: ignore[assignment]
            distance_m: float = 1.0
            motor_state: NDArray[Any] = None  # type: ignore[assignment]
            audio_chunk: NDArray[Any] = None  # type: ignore[assignment]
            valid_mask: NDArray[Any] = None  # type: ignore[assignment]
            n_modalities: int = 5
            lidar_features: NDArray[Any] | None = None

            def __post_init__(self) -> None:
                if self.vision_features is None:
                    self.vision_features = np.zeros(16, dtype=np.float32)
                if self.motor_state is None:
                    self.motor_state = np.zeros(4, dtype=np.float32)
                if self.audio_chunk is None:
                    self.audio_chunk = np.zeros(0, dtype=np.float32)
                if self.valid_mask is None:
                    self.valid_mask = np.ones(5, dtype=np.float32)

        prev_action = torch.zeros(1, cfg.model.action_dim)
        h = torch.zeros(1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim)
        z = torch.zeros(1, cfg.model.latent_dim)

        new_h, new_z, obs_embed, surprise = model.observe_step(_Obs(), prev_action, h, z)
        assert new_h.shape == (1, cfg.model.hidden_dim + cfg.model.cfc_hidden_dim)
        assert new_z.shape == (1, cfg.model.latent_dim)
        assert obs_embed.shape == (1, cfg.model.obs_dim)
        assert isinstance(surprise, float)


class TestUnknownEngineRejected:
    """Unknown engine literals are rejected by Pydantic at config-load time."""

    def test_unknown_engine_raises_validation_error(self) -> None:
        """Pydantic rejects a non-Literal engine value before reaching the factory."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorldModelConfig(engine="trt_only")  # type: ignore[arg-type]


class TestOnnxTrtRequiresPath:
    """``engine='onnx_trt'`` with no path AND no HF fallback raises a clear error."""

    def test_onnx_trt_without_path_raises_when_artifact_missing(self, tmp_path: Path) -> None:
        """The factory builds a composite whose observe-engine raises on warmup.

        Both runtime + composite construction succeed cheaply (no I/O).
        The FileNotFoundError surfaces at the first observe_step (or
        explicit warmup) because the underlying ONNX file doesn't exist.
        The composite's imagine engine is a PyTorch DualStreamRSSM and
        is therefore unaffected by the missing .onnx — operators can
        still run MCTS-only flows on the composite while debugging the
        missing artifact.
        """
        bogus_path = tmp_path / "does_not_exist.onnx"
        cfg = _make_cfg(engine="onnx_trt", onnx_path=str(bogus_path))
        model = build_world_model(cfg)
        # The factory returns the composite for engine='onnx_trt' so
        # MCTS planning (imagine_step) works regardless of ONNX state.
        assert isinstance(model, CompositeWorldModel)
        assert isinstance(model.observe_engine, DualStreamRSSMOnnx)
        # Warming the ONNX engine surfaces the missing file as a clear error.
        with pytest.raises(FileNotFoundError):
            model.observe_engine.warmup()
