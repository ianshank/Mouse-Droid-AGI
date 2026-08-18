"""Integration test for ``scripts/spike_step_distillation.py`` (F-023 spike).

Tiny in-process run: finite losses, stable report schema, deterministic under
a fixed seed (the teacher is prior-MEAN, so eval MSE/agreement are exact
across runs), paramless-teacher contract, and no hardware/mujoco deps.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def spike() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "spike_step_distillation",
        _REPO_ROOT / "scripts" / "spike_step_distillation.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TINY_ARGS = [
    "--k",
    "2",
    "--distill-steps",
    "5",
    "--batch-size",
    "16",
    "--n-states",
    "32",
    "--trials",
    "10",
    "--seed",
    "42",
]


def test_tiny_run_writes_stable_finite_report(spike: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "spike.json"
    # Snapshot sys.modules BEFORE main() — prior tests may have imported mujoco
    # via pytest.importorskip. The contract is no NEW import, not none ever.
    pre_modules = frozenset(sys.modules)
    rc = spike.main([*_TINY_ARGS, "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    # Device-agnostic: "container-cpu" here, "container-cuda" on a GPU host.
    assert report["environment"].startswith("container-")
    assert report["device"] in ("cpu", "cuda", "cuda:0")
    assert report["consumer_ceiling"]["end_to_end_ceiling"] == "~1.25-1.6x"
    (row,) = report["results"]
    assert row["k"] == 2
    assert math.isfinite(row["distill_loss_last"])
    assert math.isfinite(row["eval_mse"]["hz"])
    assert math.isfinite(row["eval_mse"]["return"])
    assert 0.0 <= row["action_agreement"] <= 1.0
    for family in ("primitive_latency_ms", "student_latency_ms"):
        for pct in ("p50", "p95", "p99"):
            assert row[family][pct] > 0.0
    new_modules = frozenset(sys.modules) - pre_modules
    assert "mujoco" not in new_modules, (
        "spike distillation imported mujoco — should stay lazy"
    )


def test_deterministic_accuracy_across_runs(spike: ModuleType, tmp_path: Path) -> None:
    """Prior-MEAN teacher ⇒ eval MSE + agreement byte-identical across runs."""
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    assert spike.main([*_TINY_ARGS, "--out", str(out_a)]) == 0
    assert spike.main([*_TINY_ARGS, "--out", str(out_b)]) == 0
    row_a = json.loads(out_a.read_text())["results"][0]
    row_b = json.loads(out_b.read_text())["results"][0]
    assert row_a["eval_mse"] == row_b["eval_mse"]
    assert row_a["action_agreement"] == row_b["action_agreement"]
    assert row_a["distill_loss_last"] == row_b["distill_loss_last"]


def test_teacher_adapter_is_paramless_and_nonmutating(spike: ModuleType) -> None:
    """The adapter must register NO parameters and never freeze the shared RSSM."""
    from mousedroid.config.schema import ModelConfig
    from mousedroid.world_model.rssm import RSSM

    mcfg = ModelConfig.model_validate(
        {
            "vision_dim": 0,
            "vision_proj_dim": 0,
            "ultrasonic_dim": 1,
            "hidden_dim": 16,
            "latent_dim": 8,
            "obs_dim": 16,
            "ultrasonic_proj_dim": 4,
            "motor_proj_dim": 8,
        }
    )
    torch.manual_seed(0)
    model = RSSM(mcfg)
    adapter = spike.KStepTeacherAdapter(model, k=2, gamma=0.97)
    assert list(adapter.parameters()) == []
    # The distiller's freeze loop over adapter.parameters() is a no-op —
    # the shared RSSM keeps requires_grad.
    for param in adapter.parameters():  # pragma: no cover - empty by contract
        param.requires_grad = False
    assert all(p.requires_grad for p in model.parameters())
    x = torch.randn(3, 16 + 8 + 2 * mcfg.action_dim)
    out = adapter(x)
    assert out.shape == (3, 16 + 8 + 1)
    # Deterministic prior-mean composition: same input ⇒ identical output.
    assert torch.equal(out, adapter(x))
