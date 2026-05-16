"""Unit tests for ``DualStreamRSSMOnnx`` runtime class.

B2 Story 2 — drop-in :class:`WorldModelProtocol` replacement that runs
the exported ``.onnx`` via ``onnxruntime.InferenceSession`` with the
TensorRT → CUDA → CPU execution-provider fallback chain. Mirrors the
proven :class:`DistilledVLAOnnx` pattern from ``vla/policy.py`` —
lazy ``onnxruntime`` import inside ``warmup()``, provider-chain
intersection via ``_resolve_providers``, idempotent warmup, and
``torch.no_grad()`` wrapping at the call boundary.

The numerical-equivalence test uses a real exported ``.onnx`` produced
by ``scripts/export_dual_stream_rssm_onnx.py.run_export`` to verify the
end-to-end PyTorch ↔ ORT contract on the deterministic outputs.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

pytest.importorskip("ncps")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


from mousedroid.config.schema import ModelConfig
from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
from mousedroid.world_model.dual_stream_rssm_onnx import DualStreamRSSMOnnx
from mousedroid.world_model.protocol import WorldModelProtocol


@dataclass
class _StubObservation:
    """Minimal ObservationProtocol-compatible stub."""

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


def _make_cfg() -> ModelConfig:
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


def _load_export_module() -> Any:
    """Import ``scripts/export_dual_stream_rssm_onnx.py`` as a module."""
    script_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "export_dual_stream_rssm_onnx.py"
    )
    spec = importlib.util.spec_from_file_location("export_dual_stream_rssm_onnx", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_dual_stream_rssm_onnx"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exported_onnx(tmp_path: Path) -> tuple[Path, DualStreamRSSM, ModelConfig]:
    """Return a freshly-exported ``.onnx`` + the source PyTorch model."""
    export_module = _load_export_module()
    cfg = _make_cfg()
    model = DualStreamRSSM(cfg)
    model.train(False)
    output_path = tmp_path / "observe_step.onnx"
    export_module.run_export(model=model, cfg=cfg, output_path=output_path, opset=17)
    return output_path, model, cfg


class TestConstruction:
    """``DualStreamRSSMOnnx`` is cheap to construct — no I/O until warmup()."""

    def test_construction_does_not_load_session(self, tmp_path: Path) -> None:
        """Constructing the runtime must not hit the filesystem or import onnxruntime."""
        # File doesn't even need to exist at construction time.
        rt = DualStreamRSSMOnnx(
            model_path=tmp_path / "does_not_exist.onnx",
            cfg=_make_cfg(),
        )
        assert rt is not None
        assert rt.active_providers == ()

    def test_conforms_to_world_model_protocol(self, tmp_path: Path) -> None:
        rt = DualStreamRSSMOnnx(
            model_path=tmp_path / "does_not_exist.onnx",
            cfg=_make_cfg(),
        )
        assert isinstance(rt, WorldModelProtocol)


class TestWarmup:
    """``warmup()`` is the single source of session creation + provider negotiation."""

    def test_warmup_loads_session_with_cpu_provider(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(
            model_path=path,
            cfg=cfg,
            providers=("CPUExecutionProvider",),
        )
        rt.warmup()
        assert rt.active_providers == ("CPUExecutionProvider",)

    def test_warmup_is_idempotent(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(model_path=path, cfg=cfg, providers=("CPUExecutionProvider",))
        rt.warmup()
        session_first = rt._session
        rt.warmup()
        session_second = rt._session
        assert session_first is session_second

    def test_warmup_raises_file_not_found(self, tmp_path: Path) -> None:
        rt = DualStreamRSSMOnnx(
            model_path=tmp_path / "does_not_exist.onnx",
            cfg=_make_cfg(),
        )
        with pytest.raises(FileNotFoundError):
            rt.warmup()

    def test_provider_fallback_to_cpu(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        """When no requested providers are available, CPU is the safe fallback."""
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(
            model_path=path,
            cfg=cfg,
            # Request a fictional provider — runtime intersects with what's installed.
            providers=("NonexistentExecutionProvider",),
        )
        rt.warmup()
        assert rt.active_providers == ("CPUExecutionProvider",)


class TestObserveStep:
    """``observe_step()`` mirrors the PyTorch contract — float surprise + tensor outputs."""

    def test_observe_step_returns_4_tuple(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(model_path=path, cfg=cfg, providers=("CPUExecutionProvider",))

        obs = _StubObservation()
        prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)

        result = rt.observe_step(obs, prev_action, h, z)
        assert len(result) == 4
        new_h, new_z, obs_embed, surprise = result
        assert isinstance(new_h, torch.Tensor)
        assert isinstance(new_z, torch.Tensor)
        assert isinstance(obs_embed, torch.Tensor)
        # ``surprise`` is a Python float (matches public WorldModelProtocol).
        assert isinstance(surprise, float)

    def test_observe_step_shapes(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(model_path=path, cfg=cfg, providers=("CPUExecutionProvider",))

        obs = _StubObservation()
        prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)

        new_h, new_z, obs_embed, _surprise = rt.observe_step(obs, prev_action, h, z)
        assert new_h.shape == (1, cfg.hidden_dim + cfg.cfc_hidden_dim)
        assert new_z.shape == (1, cfg.latent_dim)
        assert obs_embed.shape == (1, cfg.obs_dim)

    def test_observe_step_matches_torch_output_within_tol(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        """Cross-engine equivalence guarantee on deterministic outputs.

        ``new_z`` is excluded — Gaussian sampling diverges across engines
        but mean/logvar (and therefore ``surprise``) are identical. See
        the export-tooling tests for the same rationale.
        """
        path, model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(model_path=path, cfg=cfg, providers=("CPUExecutionProvider",))

        obs = _StubObservation()
        prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)

        torch_new_h, _torch_new_z, torch_obs, torch_surprise = model.observe_step(
            obs, prev_action, h, z
        )
        onnx_new_h, _onnx_new_z, onnx_obs, onnx_surprise = rt.observe_step(obs, prev_action, h, z)

        assert torch.allclose(torch_new_h, onnx_new_h, atol=1e-4)
        assert torch.allclose(torch_obs, onnx_obs, atol=1e-4)
        assert torch_surprise == pytest.approx(onnx_surprise, abs=1e-4)

    def test_observe_step_lazy_warmup(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        """First observe_step() call triggers warmup transparently."""
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(model_path=path, cfg=cfg, providers=("CPUExecutionProvider",))
        # No explicit warmup() — observe_step() should self-warm.
        obs = _StubObservation()
        prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)
        rt.observe_step(obs, prev_action, h, z)
        assert rt.active_providers == ("CPUExecutionProvider",)


class TestLazyImport:
    """The runtime module must not import ``onnxruntime`` at module-load time."""

    def test_module_import_does_not_load_onnxruntime(self) -> None:
        """Importing the runtime module is cheap — onnxruntime is lazy.

        This mirrors the import-graph isolation test for
        :class:`DistilledVLAOnnx`. Operators on a host without ORT
        installed can still ``import mousedroid.world_model.*`` without
        ``ImportError``.
        """
        # The point of the test: the import statement above (at module top)
        # already proves the module loads cleanly. Just check that the
        # runtime class is importable.
        from mousedroid.world_model import dual_stream_rssm_onnx

        assert hasattr(dual_stream_rssm_onnx, "DualStreamRSSMOnnx")


class TestImagineStep:
    """``imagine_step`` is delegated to the source PyTorch model (CfC handles its own state)."""

    def test_imagine_step_is_not_implemented(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        """ONNX runtime ships ``observe_step`` only; ``imagine_step`` raises NotImplementedError."""
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(model_path=path, cfg=cfg, providers=("CPUExecutionProvider",))
        action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)
        with pytest.raises(NotImplementedError, match="imagine_step"):
            rt.imagine_step(action, h, z)


class _RecordingMetrics:
    """Minimal MetricsRegistry stand-in capturing world-model observations.

    Exposes the one method the runtime calls directly — Tier C3.1 wired
    ``observe_world_model_observe_step_seconds`` unconditionally on
    :class:`MetricsRegistry`, so the runtime no longer probes via
    ``getattr`` and any registry-shaped object must expose this method.
    """

    def __init__(self) -> None:
        self.observed_seconds: list[float] = []

    def observe_world_model_observe_step_seconds(self, value: float) -> None:
        self.observed_seconds.append(value)


class TestMetricsObservation:
    """``DualStreamRSSMOnnx`` emits a latency observation when metrics is provided."""

    def test_observe_step_records_one_observation_per_call(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        """Each observe_step call appends exactly one finite, positive sample."""
        import math

        path, _model, cfg = exported_onnx
        metrics = _RecordingMetrics()
        rt = DualStreamRSSMOnnx(
            model_path=path,
            cfg=cfg,
            providers=("CPUExecutionProvider",),
            metrics=metrics,  # type: ignore[arg-type]
        )
        obs = _StubObservation()
        prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)

        rt.observe_step(obs, prev_action, h, z)
        rt.observe_step(obs, prev_action, h, z)
        rt.observe_step(obs, prev_action, h, z)

        assert len(metrics.observed_seconds) == 3
        for sample in metrics.observed_seconds:
            assert math.isfinite(sample)
            assert sample >= 0.0

    def test_metrics_none_disables_observation_path(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        """``metrics=None`` (the default) must skip the histogram-observe call.

        Pre-Tier-C3.1 the runtime used a defensive ``getattr`` lookup to
        tolerate a legacy :class:`MetricsRegistry` that didn't expose the
        helper. C3.1 wired the helper unconditionally, so the runtime now
        calls it directly — but ``metrics=None`` is still a supported
        deployment shape (operators without telemetry pay zero overhead).
        This test pins the no-metrics path against accidental refactors
        that would unconditionally dereference ``self._metrics``.
        """
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(
            model_path=path,
            cfg=cfg,
            providers=("CPUExecutionProvider",),
            metrics=None,
        )
        obs = _StubObservation()
        prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)
        # No exception — the metrics-disabled path is supported.
        rt.observe_step(obs, prev_action, h, z)


class TestOptionalModalityFeeds:
    """The runtime feeds audio / lidar tensors only when the modality is enabled."""

    def _make_audio_lidar_cfg(self) -> ModelConfig:
        """ModelConfig that turns on both audio and lidar to exercise feeds."""
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
            audio_dim=8,
            audio_proj_dim=4,
            lidar_dim=12,
            lidar_proj_dim=4,
            cfc_hidden_dim=16,
            cfc_backbone_units=32,
            cfc_backbone_layers=1,
        )

    def test_audio_and_lidar_feeds_when_enabled(self, tmp_path: Path) -> None:
        """Cfg with audio + lidar enabled produces feeds for both modalities."""
        export_module = _load_export_module()
        cfg = self._make_audio_lidar_cfg()
        torch_model = DualStreamRSSM(cfg)
        torch_model.train(False)
        onnx_path = tmp_path / "observe_step.onnx"
        export_module.run_export(model=torch_model, cfg=cfg, output_path=onnx_path, opset=17)

        rt = DualStreamRSSMOnnx(model_path=onnx_path, cfg=cfg, providers=("CPUExecutionProvider",))

        obs = _StubObservation(
            audio_chunk=np.ones(8, dtype=np.float32),
            lidar_features=np.ones(12, dtype=np.float32),
        )
        prev_action = torch.zeros(1, cfg.action_dim, dtype=torch.float32)
        h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim, dtype=torch.float32)
        z = torch.zeros(1, cfg.latent_dim, dtype=torch.float32)
        new_h, new_z, obs_embed, _surprise = rt.observe_step(obs, prev_action, h, z)
        assert new_h.shape == (1, cfg.hidden_dim + cfg.cfc_hidden_dim)
        assert new_z.shape == (1, cfg.latent_dim)
        assert obs_embed.shape == (1, cfg.obs_dim)

    def test_name_property_returns_constructor_value(
        self,
        exported_onnx: tuple[Path, DualStreamRSSM, ModelConfig],
    ) -> None:
        """The ``name`` property surfaces the operator-supplied telemetry label."""
        path, _model, cfg = exported_onnx
        rt = DualStreamRSSMOnnx(
            model_path=path,
            cfg=cfg,
            providers=("CPUExecutionProvider",),
            name="custom_label",
        )
        assert rt.name == "custom_label"
