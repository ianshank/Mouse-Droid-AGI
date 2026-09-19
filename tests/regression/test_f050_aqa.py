"""AQA — F-050 observe-step instrumentation seam.

Pins the behaviour this feature adds, so it cannot silently regress:

* ``build_world_model`` accepts a metrics registry and hands it to the engine.
* Both PyTorch engines report into
  ``mousedroid_world_model_observe_step_seconds``, so the family finally has a
  production writer and ``WorldModelObserveStepLatencyHigh`` can evaluate.
* The timing boundary records on success only (telemetry invariant 5).
* No new histogram bucket literal leaks into code — boundaries stay in config.

Torch-only by design: the blocking ``test`` job installs ``.[dev,telemetry,mcp]``
with no onnxruntime, and ``scripts/validations/F-050.sh`` fails on a skip-all
collection, so every assertion here must run without ORT.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from mousedroid.config.schema import MetricsConfig, ModelConfig, Settings
from mousedroid.factory.world_model import build_world_model
from mousedroid.telemetry.metrics.registry import MetricsRegistry
from mousedroid.world_model.observe_step_timing import (
    ObserveStepLatencySink,
    observe_step_latency,
)
from mousedroid.world_model.rssm import RSSM

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAMILY = "mousedroid_world_model_observe_step_seconds"


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
        cfc_hidden_dim=0,
    )


def _count(registry: MetricsRegistry) -> int | None:
    for line in registry.render_prometheus().splitlines():
        if line.startswith(f"{_FAMILY}_count"):
            return int(float(line.split()[-1]))
    return None


def test_build_world_model_exposes_a_keyword_only_metrics_seam() -> None:
    parameter = inspect.signature(build_world_model).parameters["metrics"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_registry_reaches_the_constructed_engine() -> None:
    settings = Settings(mock_hardware=True, model=_cfg())
    registry = MetricsRegistry(settings.metrics)
    model = build_world_model(settings, metrics=registry)
    assert model._metrics is registry  # type: ignore[attr-defined]


def test_torch_engine_populates_the_observe_step_histogram() -> None:
    """The regression this feature fixes: the family had no production writer."""
    cfg = _cfg()
    settings = Settings(mock_hardware=True, model=cfg)
    registry = MetricsRegistry(settings.metrics)
    model = build_world_model(settings, metrics=registry)

    h = torch.zeros(1, cfg.hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)
    action = torch.zeros(1, cfg.action_dim)
    for _ in range(2):
        h, z, _, _ = model.observe_step(_Obs(), action, h, z)

    assert _count(registry) == 2


def test_timing_records_on_success_only() -> None:
    """telemetry/CLAUDE.md invariant 5 — never record on an error path."""
    recorded: list[float] = []

    class _Sink:
        def observe_world_model_observe_step_seconds(self, value: float) -> None:
            recorded.append(value)

    try:
        with observe_step_latency(_Sink(), engine="aqa"):
            msg = "inference failed"
            raise ValueError(msg)
    except ValueError:
        pass
    assert recorded == []


def test_metrics_registry_satisfies_the_latency_sink_protocol() -> None:
    """Engines depend on the Protocol, not the concrete registry."""
    registry = MetricsRegistry(Settings(mock_hardware=True).metrics)
    assert isinstance(registry, ObserveStepLatencySink)


def test_both_torch_engines_accept_the_metrics_keyword() -> None:
    """DualStreamRSSM is checked by signature so ncps stays optional here."""
    from mousedroid.world_model import dual_stream_rssm as dual_stream_module

    for target in (RSSM.__init__, dual_stream_module.DualStreamRSSM.__init__):
        parameter = inspect.signature(target).parameters["metrics"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_orchestrator_factory_builds_the_registry_before_the_world_model() -> None:
    """Ordering is the whole fix — a later registry cannot reach the engine."""
    source = (_REPO_ROOT / "src" / "mousedroid" / "factory" / "orchestrator.py").read_text(
        encoding="utf-8"
    )
    registry_at = source.index("metrics_registry = build_metrics_registry(cfg)")
    world_model_at = source.index("wm = build_world_model(")
    assert registry_at < world_model_at, (
        "build_metrics_registry must precede build_world_model; otherwise the "
        "engine is constructed before a writer exists"
    )
    assert source.count("metrics_registry = build_metrics_registry(cfg)") == 1, (
        "the registry must be built exactly once — two registries would split "
        "the exposition between one that is served and one that is not"
    )


def test_histogram_buckets_stay_in_config_not_code() -> None:
    """CLAUDE.md invariant 2 — no threshold literals in runtime code."""
    assert "world_model_observe_step_seconds_buckets" in MetricsConfig.model_fields
    helper = (
        _REPO_ROOT / "src" / "mousedroid" / "world_model" / "observe_step_timing.py"
    ).read_text(encoding="utf-8")
    assert "0.001" not in helper, "bucket literals belong in MetricsConfig, not the helper"
    assert "0.033" not in helper, "deadline literals belong in config, not the helper"


def test_timing_helper_uses_no_assert_statements() -> None:
    """ruff S101 is blocking in src/; PYTHONOPTIMIZE=1 strips asserts on the rover."""
    helper = (
        _REPO_ROOT / "src" / "mousedroid" / "world_model" / "observe_step_timing.py"
    ).read_text(encoding="utf-8")
    for line in helper.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("assert "), f"assert in src/: {stripped}"
