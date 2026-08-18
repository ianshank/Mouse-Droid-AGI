"""Integration test for ``scripts/compare_drift.py`` (F-023).

Runs ``main()`` in-process with tiny synthetic sizes: finite JSON report,
exit 0 report-only, exit 1 under an impossible ``--gate-max-regression``, and
no mujoco import on the default path.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compare_drift() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "compare_drift", _REPO_ROOT / "scripts" / "compare_drift.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TINY_ARGS = [
    "--episodes",
    "2",
    "--seq-len",
    "16",
    "--steps",
    "3",
    "--train-batches",
    "2",
    "--context-steps",
    "3",
    "--horizon",
    "6",
    "--seed",
    "42",
]


def test_report_only_run_writes_finite_json(compare_drift: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "drift.json"
    # Snapshot sys.modules BEFORE main() — prior tests (e.g. test_sim_episode_generator)
    # may have already imported mujoco via pytest.importorskip. The contract is that the
    # synthetic drift path does not ADD a mujoco import, not that no prior test loaded it.
    pre_modules = frozenset(sys.modules)
    rc = compare_drift.main([*_TINY_ARGS, "--memory", "both", "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["source"] == "synthetic"
    assert report["baseline"]["headline_channel"] == "range"
    for arm in ("baseline", "augmented"):
        for value in report[arm]["means"].values():
            assert math.isfinite(value)
    ablation = report["memory_ablation"]
    assert math.isfinite(ablation["memory_on"]["means"]["latent_h"])
    # The default synthetic path must never import mujoco itself.
    new_modules = frozenset(sys.modules) - pre_modules
    assert "mujoco" not in new_modules, (
        "synthetic drift path imported mujoco — should stay lazy"
    )


def test_gate_max_regression_can_fail(compare_drift: ModuleType, tmp_path: Path) -> None:
    """A negative gate threshold makes the ceiling < baseline — must exit 1."""
    out = tmp_path / "gated.json"
    rc = compare_drift.main([*_TINY_ARGS, "--out", str(out), "--gate-max-regression", "-0.9999"])
    assert rc == 1


def test_deterministic_across_runs(compare_drift: ModuleType, tmp_path: Path) -> None:
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    assert compare_drift.main([*_TINY_ARGS, "--out", str(out_a)]) == 0
    assert compare_drift.main([*_TINY_ARGS, "--out", str(out_b)]) == 0
    assert json.loads(out_a.read_text()) == json.loads(out_b.read_text())
