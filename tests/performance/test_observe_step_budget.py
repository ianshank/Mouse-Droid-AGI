"""Performance budget regression for ``DualStreamRSSM.observe_step``.

Tier B Track B2 Story 4 — slow-marked perf gate that asserts both the
PyTorch and ONNX engines complete one ``observe_step`` within an
operator-tunable wall-clock budget.

Budget contract
---------------
``MOUSEDROID_OBSERVE_STEP_BUDGET_MS`` env var sets the mean wall-clock
budget per ``observe_step`` invocation (over ``_ITERATIONS`` samples).
The default is **33 ms** — the 30 Hz orchestrator tick — which is the
portable bound that any dev workstation (Mac M-series, x86 with or
without GPU) should meet. The Jetson Orin Nano production gate runs
with ``MOUSEDROID_OBSERVE_STEP_BUDGET_MS=10`` — the <10ms target this
sprint sets to keep the world-model hot path under one third of the
control loop budget.

Skip semantics
--------------
The "onnx_trt" parametrisation skips cleanly when:
- ``onnxruntime`` is not installed (default ``[dev]`` install path)
- The exported ``.onnx`` artifact is not available (this PR's library
  entry point produces one in-process, so the test runs OK from a
  fresh checkout; production CI invokes the export script first).

Mirrors :file:`tests/performance/test_offline_rl_bc_overhead.py` for the
env-var-tunable budget pattern (PR-A1 precedent).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

pytest.importorskip("ncps")


_DEFAULT_BUDGET_MS: float = 33.0
"""Portable dev budget — one 30Hz orchestrator tick."""

_BUDGET_ENV: str = "MOUSEDROID_OBSERVE_STEP_BUDGET_MS"
_ITERATIONS: int = 50


def _resolve_budget_ms() -> float:
    """Read the budget from env, default to 33ms (30Hz tick)."""
    raw = os.environ.get(_BUDGET_ENV)
    if raw is None:
        return _DEFAULT_BUDGET_MS
    parsed = float(raw)
    if parsed <= 0.0:
        msg = f"{_BUDGET_ENV} must be > 0.0 (got {parsed!r})"
        raise ValueError(msg)
    return parsed


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


def _make_cfg() -> Any:
    from mousedroid.config.schema import ModelConfig

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


@dataclass
class _Obs:
    """Minimal ObservationProtocol-compatible stub for perf runs."""

    timestamp: float = 0.0
    vision_features: NDArray[np.float32] | None = None
    distance_m: float = 1.0
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


@pytest.mark.slow
def test_observe_step_torch_engine_within_budget() -> None:
    """PyTorch engine observe_step mean latency stays within the budget."""
    from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM

    cfg = _make_cfg()
    model = DualStreamRSSM(cfg)
    model.train(False)

    obs = _Obs()
    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)

    # Warm one call (avoid first-call JIT/cache overhead from biasing the mean).
    model.observe_step(obs, prev_action, h, z)

    started = time.perf_counter()
    for _ in range(_ITERATIONS):
        model.observe_step(obs, prev_action, h, z)
    elapsed_s = time.perf_counter() - started
    mean_ms = (elapsed_s / _ITERATIONS) * 1000.0

    budget_ms = _resolve_budget_ms()
    assert mean_ms < budget_ms, (
        f"PyTorch observe_step mean latency {mean_ms:.2f}ms exceeds "
        f"{_BUDGET_ENV}={budget_ms:.2f}ms over {_ITERATIONS} iterations. "
        f"Set {_BUDGET_ENV} higher in CI env for slow hardware."
    )


@pytest.mark.slow
def test_observe_step_onnx_engine_within_budget(tmp_path: Path) -> None:
    """ONNX runtime observe_step mean latency stays within the budget."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from mousedroid.world_model.dual_stream_rssm import DualStreamRSSM
    from mousedroid.world_model.dual_stream_rssm_onnx import DualStreamRSSMOnnx

    export_module = _load_export_module()
    cfg = _make_cfg()
    torch_model = DualStreamRSSM(cfg)
    torch_model.train(False)
    onnx_path = tmp_path / "observe_step.onnx"
    export_module.run_export(model=torch_model, cfg=cfg, output_path=onnx_path, opset=17)

    rt = DualStreamRSSMOnnx(model_path=onnx_path, cfg=cfg, providers=("CPUExecutionProvider",))
    rt.warmup()

    obs = _Obs()
    prev_action = torch.zeros(1, cfg.action_dim)
    h = torch.zeros(1, cfg.hidden_dim + cfg.cfc_hidden_dim)
    z = torch.zeros(1, cfg.latent_dim)

    rt.observe_step(obs, prev_action, h, z)

    started = time.perf_counter()
    for _ in range(_ITERATIONS):
        rt.observe_step(obs, prev_action, h, z)
    elapsed_s = time.perf_counter() - started
    mean_ms = (elapsed_s / _ITERATIONS) * 1000.0

    budget_ms = _resolve_budget_ms()
    assert mean_ms < budget_ms, (
        f"ONNX observe_step mean latency {mean_ms:.2f}ms exceeds "
        f"{_BUDGET_ENV}={budget_ms:.2f}ms over {_ITERATIONS} iterations. "
        f"On Jetson the 10ms target may require TensorRT EP; CPU EP is "
        f"a portable fallback that may not meet the production budget."
    )


@pytest.mark.slow
def test_budget_env_validates_positive() -> None:
    """_resolve_budget_ms rejects non-positive overrides — operator safety net."""
    saved = os.environ.get(_BUDGET_ENV)
    try:
        os.environ[_BUDGET_ENV] = "-5"
        with pytest.raises(ValueError, match=_BUDGET_ENV):
            _resolve_budget_ms()
    finally:
        if saved is None:
            os.environ.pop(_BUDGET_ENV, None)
        else:
            os.environ[_BUDGET_ENV] = saved
