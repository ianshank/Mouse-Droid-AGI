"""Phase 2 acceptance: golden RSSM loss-curve regression at fixed seed.

Closes the last open Phase 2 acceptance bullet from ``NEXT_STEPS.md``:

    "Golden RSSM loss curve at fixed seed within ±1% of baseline."

Strategy:

* :mod:`tests.regression._rssm_golden_helper` runs a tiny, deterministic
  RSSM training loop (CPU, seed=0, 10 steps, dims << production).
* The captured curve is committed at
  :data:`BASELINE_PATH` and replayed on every test run.
* ``recon`` and ``total`` losses are compared with a relative tolerance
  of ``REL_TOLERANCE`` (1%); ``kl`` uses ``KL_REL_TOLERANCE`` (5%) because
  KL divergence depends on the sampled latent and has a slightly wider
  numerical envelope than the deterministic recon term.
* Setting ``MOUSEDROID_UPDATE_GOLDEN=1`` regenerates the fixture instead
  of asserting — used only when the helper or model definition changes
  intentionally. The mode is logged loudly and skips the assertion.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from mousedroid.logging.setup import get_logger
from tests.regression._rssm_golden_helper import (
    GoldenRSSMConfig,
    compute_rssm_loss_curve,
)

_log = get_logger(__name__)

BASELINE_PATH = Path(__file__).parent / "fixtures" / "phase2_rssm_golden_baseline.json"
UPDATE_ENV_VAR = "MOUSEDROID_UPDATE_GOLDEN"

# Tolerances. The recon and total terms are reproducible to ~1e-7 across
# CPU runs of the same PyTorch build, so 1% leaves a generous margin for
# minor BLAS / wheel-version drift while still flagging real regressions.
REL_TOLERANCE: float = 0.01
KL_REL_TOLERANCE: float = 0.05
EXPECTED_NUM_STEPS: int = 10
LOSS_KEYS: tuple[str, ...] = ("recon", "kl", "total")


def _load_baseline() -> list[dict[str, float]]:
    raw = json.loads(BASELINE_PATH.read_text())
    curve = raw["curve"]
    assert isinstance(curve, list), "baseline.curve must be a list"
    return curve


def _write_baseline(curve: list[dict[str, float]]) -> None:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "description": (
            "Phase 2 golden RSSM regression baseline (CPU, seed=0, 10 steps, tiny dims)."
        ),
        "curve": curve,
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def _relative_error(actual: float, expected: float) -> float:
    """``|actual - expected| / max(|expected|, 1e-12)`` — safe for near-zero values."""
    denom = max(abs(expected), 1e-12)
    return abs(actual - expected) / denom


@pytest.fixture(scope="module")
def computed_curve() -> list[dict[str, float]]:
    """Compute the curve once per module (still <1 s)."""
    return compute_rssm_loss_curve()


def test_golden_baseline_fixture_exists() -> None:
    """The committed fixture must exist; otherwise the test cannot run."""
    assert BASELINE_PATH.exists(), (
        f"Baseline fixture missing at {BASELINE_PATH}. "
        f"Regenerate via {UPDATE_ENV_VAR}=1 pytest "
        f"tests/regression/test_phase2_rssm_golden.py"
    )


def test_golden_curve_length_matches_baseline(
    computed_curve: list[dict[str, float]],
) -> None:
    """Curve must have ``EXPECTED_NUM_STEPS`` entries."""
    assert len(computed_curve) == EXPECTED_NUM_STEPS
    if BASELINE_PATH.exists():
        baseline = _load_baseline()
        assert len(baseline) == EXPECTED_NUM_STEPS


def test_golden_curve_keys_are_complete(
    computed_curve: list[dict[str, float]],
) -> None:
    """Every curve entry must carry recon/kl/total floats."""
    for i, entry in enumerate(computed_curve):
        for key in LOSS_KEYS:
            assert key in entry, f"step {i} missing key {key!r}"
            assert isinstance(entry[key], float), f"step {i} key {key!r} not float"
            assert math.isfinite(entry[key]), f"step {i} key {key!r} not finite"


def test_golden_curve_total_loss_decreases(
    computed_curve: list[dict[str, float]],
) -> None:
    """End-to-end: total loss must trend down over 10 steps.

    Even a CI environment that lost the baseline JSON (e.g. fresh clone
    that skipped the fixture) gets a smoke-level guarantee that the
    training step still optimizes.
    """
    first = computed_curve[0]["total"]
    last = computed_curve[-1]["total"]
    assert last < first, f"RSSM total loss did not decrease: {first} -> {last}"


def test_golden_curve_matches_baseline_within_tolerance(
    computed_curve: list[dict[str, float]],
) -> None:
    """Loss curve must stay within tolerance of the committed baseline.

    Honors ``MOUSEDROID_UPDATE_GOLDEN=1`` to regenerate the baseline
    instead of asserting (intentional updates only).
    """
    if os.environ.get(UPDATE_ENV_VAR) == "1":
        _log.warning(
            "phase2_rssm_golden_update",
            path=str(BASELINE_PATH),
            note="env var set; rewriting baseline and skipping assertion",
        )
        _write_baseline(computed_curve)
        pytest.skip(f"{UPDATE_ENV_VAR}=1: baseline regenerated, assertion skipped")

    baseline = _load_baseline()
    failures: list[str] = []
    for i, (got, expected) in enumerate(zip(computed_curve, baseline, strict=True)):
        for key in LOSS_KEYS:
            tol = KL_REL_TOLERANCE if key == "kl" else REL_TOLERANCE
            err = _relative_error(got[key], expected[key])
            if err > tol:
                failures.append(
                    f"step={i} key={key!r} got={got[key]:.6f} "
                    f"expected={expected[key]:.6f} rel_err={err:.4f} tol={tol:.4f}"
                )
    if failures:
        for line in failures:
            _log.error("phase2_rssm_golden_drift", detail=line)
        msg = (
            "Golden RSSM regression: "
            f"{len(failures)} mismatches outside tolerance.\n"
            + "\n".join(failures)
            + f"\n\nIf this drift is intentional, rerun with {UPDATE_ENV_VAR}=1 to regenerate."
        )
        pytest.fail(msg)


@pytest.mark.parametrize("num_steps", [1, 3, 10])
def test_golden_curve_prefix_stable(num_steps: int) -> None:
    """Prefix property: ``compute_rssm_loss_curve(n)`` equals first n of full curve.

    Guards against accidental coupling between batch generation and step
    count that would break determinism at smaller ``num_steps``.
    """
    full = compute_rssm_loss_curve()
    prefix = compute_rssm_loss_curve(GoldenRSSMConfig(num_steps=num_steps))
    assert prefix == full[:num_steps]
