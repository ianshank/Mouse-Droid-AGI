"""PR-A2.1 — performance budget regression for writer-side instrumentation.

Asserts the per-call overhead of metric emission stays within an
operator-tunable budget. Mirrors the PR-A1 pattern in
``tests/performance/test_offline_rl_bc_overhead.py``.

Marked ``slow`` so the default ``pytest`` invocation skips it;
``scripts/ci.sh`` performance stage picks it up. The budget is tunable
via ``MOUSEDROID_INSTRUMENTATION_OVERHEAD_BUDGET`` so Jetson runs
(slower per-step) can relax the bound without touching code.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from mousedroid.config.schema import MetricsConfig, VLMProgressConfig
from mousedroid.telemetry.metrics import MetricsRegistry

_ITERATIONS = 500
_DEFAULT_BUDGET = 1.15  # 15% headroom over the no-metrics baseline
_BUDGET_ENV = "MOUSEDROID_INSTRUMENTATION_OVERHEAD_BUDGET"


def _resolve_budget() -> float:
    """Return the operator-tuned overhead budget multiplier."""
    raw = os.environ.get(_BUDGET_ENV)
    if raw is None:
        return _DEFAULT_BUDGET
    parsed = float(raw)
    if parsed <= 1.0:
        msg = f"{_BUDGET_ENV} must be > 1.0 (got {parsed!r})"
        raise ValueError(msg)
    return parsed


@pytest.mark.slow
def test_mock_vla_instrumentation_within_budget() -> None:
    """MockVLA.predict() with metrics enabled must stay within budget multiplier."""
    from mousedroid.vla.policy import MockVLA, VLAObservation

    obs = VLAObservation(h=torch.zeros(1, 4), z=torch.zeros(1, 4))

    # Baseline: no metrics
    no_metrics = MockVLA(action_dim=3)
    no_metrics.predict(obs)  # warmup
    t0 = time.perf_counter()
    for _ in range(_ITERATIONS):
        no_metrics.predict(obs)
    baseline = time.perf_counter() - t0

    # Instrumented
    registry = MetricsRegistry(MetricsConfig())
    with_metrics = MockVLA(action_dim=3, metrics=registry)
    with_metrics.predict(obs)  # warmup
    t0 = time.perf_counter()
    for _ in range(_ITERATIONS):
        with_metrics.predict(obs)
    instrumented = time.perf_counter() - t0

    budget = _resolve_budget()
    ratio = instrumented / max(baseline, 1e-9)
    assert ratio <= budget, (
        f"MockVLA instrumented predict() is {ratio:.2f}x baseline "
        f"(budget {budget:.2f}x). baseline={baseline:.4f}s, "
        f"instrumented={instrumented:.4f}s. Tune via env {_BUDGET_ENV} "
        f"on slower hardware (e.g. Jetson Orin Nano)."
    )


@pytest.mark.slow
def test_vlm_progress_instrumentation_within_budget() -> None:
    """VLMProgressHead.score() with metrics enabled stays within budget.

    Drives the identity-cache hit path repeatedly (same tensor objects)
    since that's the hot-path that production workloads exercise most:
    RL rollouts re-use observation buffers across ticks.
    """
    from mousedroid.reward.vlm_progress import VLMProgressHead
    from tests.unit.reward.test_vlm_progress import _CountingBackend

    o1 = torch.zeros(1, 4)
    o2 = torch.ones(1, 4)

    baseline_head = VLMProgressHead(
        VLMProgressConfig(cache_size=128),
        backend=_CountingBackend(),
    )
    baseline_head.score(o1, o2, instruction="warmup")  # warmup → miss
    t0 = time.perf_counter()
    for _ in range(_ITERATIONS):
        baseline_head.score(o1, o2, instruction="warmup")  # all identity hits
    baseline = time.perf_counter() - t0

    registry = MetricsRegistry(MetricsConfig())
    instrumented_head = VLMProgressHead(
        VLMProgressConfig(cache_size=128),
        backend=_CountingBackend(),
        metrics=registry,
    )
    instrumented_head.score(o1, o2, instruction="warmup")  # warmup → miss
    t0 = time.perf_counter()
    for _ in range(_ITERATIONS):
        instrumented_head.score(o1, o2, instruction="warmup")  # all identity hits
    instrumented = time.perf_counter() - t0

    budget = _resolve_budget()
    ratio = instrumented / max(baseline, 1e-9)
    assert ratio <= budget, (
        f"VLMProgressHead instrumented score() is {ratio:.2f}x baseline "
        f"(budget {budget:.2f}x). baseline={baseline:.4f}s, "
        f"instrumented={instrumented:.4f}s. Tune via env {_BUDGET_ENV}."
    )
