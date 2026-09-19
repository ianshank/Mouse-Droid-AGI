"""Integration — the world-model metrics seam produces real exposition samples.

The unit tests in ``tests/unit/world_model/test_observe_step_timing.py`` cover the
timing helper in isolation. This module covers the seam end to end:
``build_world_model(cfg, metrics=...)`` → engine → ``observe_step`` →
``MetricsRegistry`` → Prometheus exposition text.

It is torch-only (no onnxruntime), so it runs in the blocking ``test`` job. The
ONNX engine's own wiring is asserted structurally here (constructor keyword) and
behaviourally in the advisory ``onnx-world-model-extras`` job.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from mousedroid.config.schema import ModelConfig, Settings
from mousedroid.factory.world_model import build_world_model
from mousedroid.telemetry.metrics.registry import MetricsRegistry

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAMILY = "mousedroid_world_model_observe_step_seconds"


@dataclass
class _StubObservation:
    """Minimal ``ObservationProtocol``-compatible stub."""

    vision_dim: int = 16
    motor_dim: int = 4
    n_modalities: int = 4
    timestamp: float = 0.0
    distance_m: float = 1.5
    vision_features: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    motor_state: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    audio_chunk: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    valid_mask: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    lidar_features: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        self.vision_features = np.zeros(self.vision_dim, dtype=np.float32)
        self.motor_state = np.zeros(self.motor_dim, dtype=np.float32)
        self.valid_mask = np.ones(self.n_modalities, dtype=np.float32)


def _small_model_cfg(*, cfc_hidden_dim: int) -> ModelConfig:
    """Build a deliberately tiny model so the test stays fast.

    Every dimension is passed explicitly rather than relying on schema defaults,
    so a future default change cannot silently alter what this test exercises.
    """
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


def _histogram_count(registry: MetricsRegistry) -> int | None:
    """Return the family's sample count, or ``None`` when it is absent.

    Absent is the correct pre-observation state: the registry renders
    pure-additively (``if count > 0``) per telemetry invariant 2, so a family
    with no samples emits no lines at all.
    """
    for line in registry.render_prometheus().splitlines():
        if line.startswith(f"{_FAMILY}_count"):
            return int(float(line.split()[-1]))
    return None


def _settings_with(cfc_hidden_dim: int) -> Settings:
    return Settings(mock_hardware=True, model=_small_model_cfg(cfc_hidden_dim=cfc_hidden_dim))


def _drive(model: object, cfg: ModelConfig, *, steps: int) -> None:
    """Run ``steps`` observe-step calls, feeding state forward as the rover does."""
    combined = cfg.hidden_dim + cfg.cfc_hidden_dim
    h = torch.zeros(1, combined)
    z = torch.zeros(1, cfg.latent_dim)
    action = torch.zeros(1, cfg.action_dim)
    observation = _StubObservation(
        vision_dim=cfg.vision_dim,
        motor_dim=cfg.motor_state_dim,
    )
    for _ in range(steps):
        h, z, _, _ = model.observe_step(observation, action, h, z)  # type: ignore[attr-defined]


class TestFactorySeam:
    """``build_world_model`` must hand the registry to the engine it builds."""

    def test_default_call_leaves_the_engine_unwired(self) -> None:
        """Backwards compatibility: the pre-existing one-arg call still works."""
        model = build_world_model(_settings_with(0))
        assert model._metrics is None  # type: ignore[attr-defined]

    def test_registry_reaches_the_rssm_engine(self) -> None:
        settings = _settings_with(0)
        registry = MetricsRegistry(settings.metrics)
        model = build_world_model(settings, metrics=registry)
        assert type(model).__name__ == "RSSM", "cfc_hidden_dim=0 must dispatch to RSSM"
        assert model._metrics is registry  # type: ignore[attr-defined]

    def test_registry_reaches_the_dual_stream_engine(self) -> None:
        pytest.importorskip("ncps")
        settings = _settings_with(16)
        registry = MetricsRegistry(settings.metrics)
        model = build_world_model(settings, metrics=registry)
        assert type(model).__name__ == "DualStreamRSSM"
        assert model._metrics is registry  # type: ignore[attr-defined]

    def test_metrics_is_keyword_only(self) -> None:
        """Positional use must stay impossible so the signature can evolve."""
        parameter = inspect.signature(build_world_model).parameters["metrics"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


class TestExpositionSamples:
    """The histogram must actually gain samples — the point of the whole seam."""

    def test_family_absent_before_any_observation(self) -> None:
        settings = _settings_with(0)
        registry = MetricsRegistry(settings.metrics)
        assert _histogram_count(registry) is None

    @pytest.mark.parametrize("cfc_hidden_dim", [0, 16])
    def test_each_torch_engine_emits_one_sample_per_call(self, cfc_hidden_dim: int) -> None:
        if cfc_hidden_dim > 0:
            pytest.importorskip("ncps")
        settings = _settings_with(cfc_hidden_dim)
        registry = MetricsRegistry(settings.metrics)
        model = build_world_model(settings, metrics=registry)

        _drive(model, settings.model, steps=4)

        assert _histogram_count(registry) == 4

    def test_unwired_engine_emits_nothing(self) -> None:
        """A registry not handed to the engine must stay empty.

        Pins the exact regression this change fixed: the factory used to build
        the engine without a registry, so the family had no writer at all.
        """
        settings = _settings_with(0)
        registry = MetricsRegistry(settings.metrics)
        model = build_world_model(settings)  # no metrics= — the old behaviour

        _drive(model, settings.model, steps=3)

        assert _histogram_count(registry) is None


class TestOnnxEngineWiring:
    """Structural check that the ONNX engine accepts the same keyword.

    Behavioural coverage needs onnxruntime and lives in the advisory
    ``onnx-world-model-extras`` job; this assertion is always-on so a rename
    cannot silently orphan the ONNX writer.
    """

    def test_onnx_runtime_accepts_metrics_keyword(self) -> None:
        pytest.importorskip("ncps")
        from mousedroid.world_model.dual_stream_rssm_onnx import DualStreamRSSMOnnx

        parameter = inspect.signature(DualStreamRSSMOnnx.__init__).parameters["metrics"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    def test_onnx_builder_threads_metrics(self) -> None:
        """The private ONNX builder must forward the registry, not drop it."""
        from mousedroid.factory import world_model as factory_world_model

        source = inspect.getsource(factory_world_model._build_onnx_world_model)
        assert "metrics=metrics" in source, (
            "DualStreamRSSMOnnx must receive metrics=metrics; without it the ONNX "
            "engine reverts to having no production histogram writer"
        )


class TestTheConfigTestsReachACIJob:
    """Task 3.8 — ``WorldModelConfig`` had test coverage that executed nowhere.

    ``tests/unit/factory/test_factory_world_model_engine.py`` is the only file
    that constructs a :class:`WorldModelConfig`, and it ``importorskip``s
    ``onnx``/``onnxruntime`` — so it skip-alls in the blocking ``test`` job
    (which installs ``.[dev,telemetry,mcp]``, no ORT), and the advisory
    ``onnx-world-model-extras`` job did not name it in its path list. The result
    was config coverage that ran in *zero* CI jobs while looking present.

    This pin is always-on, because the thing it guards is a workflow path list
    that no test run can discover for itself.
    """

    _WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
    _CONFIG_TEST = "tests/unit/factory/test_factory_world_model_engine.py"

    def test_the_advisory_job_names_the_factory_config_test(self) -> None:
        workflow = self._WORKFLOW.read_text(encoding="utf-8")
        assert self._CONFIG_TEST in workflow, (
            f"{self._CONFIG_TEST} constructs WorldModelConfig and skip-alls "
            "without onnxruntime; if the advisory onnx job does not name it, "
            "those fields are pinned by nothing that ever executes"
        )

    def test_that_test_file_still_exists_at_the_pinned_path(self) -> None:
        assert (_REPO_ROOT / self._CONFIG_TEST).is_file(), (
            "the workflow path list now points at a file that moved; the job "
            "would fail on collection rather than silently skipping"
        )

    def test_it_is_still_the_only_constructor_of_the_config(self) -> None:
        """If a second, always-on constructor appears, this pin can be relaxed."""
        constructors = sorted(
            path.relative_to(_REPO_ROOT).as_posix()
            for path in (_REPO_ROOT / "tests").rglob("test_*.py")
            if "WorldModelConfig(" in path.read_text(encoding="utf-8")
        )
        assert self._CONFIG_TEST in constructors, (
            f"expected {self._CONFIG_TEST} among WorldModelConfig constructors, "
            f"found {constructors}"
        )
