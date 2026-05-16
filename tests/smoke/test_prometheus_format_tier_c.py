"""Tier C dashboard E2E smoke — Prometheus format for net-new metric families.

Mirrors :file:`tests/smoke/test_prometheus_format.py` (Tier A baseline) but
focuses on the metric families introduced by Tier B2 / Tier C work. Lives
in a separate file so the existing baseline test stays narrowly scoped to
the pre-Tier-C surface — keeping the regression net diff-readable.

What this file pins:

1. **Tier B2 world-model `observe_step` histogram** — wired by Tier C3.1
   (the registry helper ``observe_world_model_observe_step_seconds`` was
   documented in the B2 plan but only got wired in C3.1 alongside the
   Grafana panel). The ``DualStreamRSSMOnnx`` runtime class used a
   defensive ``getattr(..., None)`` lookup at
   ``world_model/dual_stream_rssm_onnx.py:293`` until this helper landed.
2. **Tier C1 cloud OTA families** (added when C1 PR lands — extensions
   below the ``# C1`` comment fail-fast if a future C1 commit forgets
   to extend ``generate_metrics_sample``).
3. **Tier C2 mission + safety families** (same pattern, extensions below
   the ``# C2`` comment).

Pattern: every new metric family declares its rendered Prometheus name in
the constants at the top of this file and asserts the name appears in
``generate_metrics_sample()`` output. ``generate_metrics_sample`` is the
single source of truth the Prometheus scrape endpoint feeds — any family
that doesn't appear there is registered-but-zero, which silently breaks
Grafana panels + alert evaluation.

Adding a new metric family in Tier C? Add it to the constants below AND
to ``generate_metrics_sample()`` in ``src/mousedroid/telemetry/metrics.py``
— the test is intentionally configured to make those two updates land
in lockstep.
"""

from __future__ import annotations

import pytest

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import generate_metrics_sample

pytestmark = pytest.mark.smoke

# ---------------------------------------------------------------------------
# Metric name derivation from config (NOT hardcoded)
# ---------------------------------------------------------------------------
_NS = MetricsConfig().namespace  # default: "mousedroid"

# Tier B2 — world-model observe_step latency histogram. Wired in Tier C3.1.
_B2_HISTOGRAM_FAMILIES: tuple[str, ...] = (f"{_NS}_world_model_observe_step_seconds",)

# Tier C1 — cloud OTA weight-update families. The C1 PR will add these to
# ``generate_metrics_sample()``. Until then the placeholders below are
# intentionally NOT enforced (commented-out fail-fast) so C3.1 can land
# without a forward dependency.
#
# C1 PR MUST uncomment and add the corresponding registry exercise calls
# in ``generate_metrics_sample()``:
#
# _C1_COUNTER_FAMILIES: tuple[str, ...] = (
#     f"{_NS}_cloud_weight_update_downloads",
#     f"{_NS}_cloud_weight_update_sha256_mismatches",
#     f"{_NS}_cloud_weight_update_swaps",
# )
# _C1_HISTOGRAM_FAMILIES: tuple[str, ...] = (
#     f"{_NS}_cloud_weight_update_download_seconds",
# )

# Tier C2 — mission + safety projection families. Same pattern as C1 —
# uncomment when C2 PR lands.
#
# _C2_COUNTER_FAMILIES: tuple[str, ...] = (
#     f"{_NS}_safety_action_clamps",
#     f"{_NS}_mission_state_transitions",
#     f"{_NS}_mission_replans",
# )
# _C2_HISTOGRAM_FAMILIES: tuple[str, ...] = (
#     f"{_NS}_mission_active_duration_seconds",
# )


def test_b2_world_model_observe_step_histogram_in_generate_sample() -> None:
    """The B2 ``observe_step`` histogram appears in ``generate_metrics_sample``.

    Before Tier C3.1 the helper ``observe_world_model_observe_step_seconds``
    was documented but not wired — the runtime class used
    ``getattr(self._metrics, "observe_world_model_observe_step_seconds", None)``
    to avoid crashing. This test pins that the helper now exists AND that
    ``generate_metrics_sample`` exercises it (rendering ``_bucket``,
    ``_sum``, and ``_count`` lines) so promtool + Grafana see non-empty
    series from the first scrape.
    """
    sample = generate_metrics_sample()
    for base_name in _B2_HISTOGRAM_FAMILIES:
        assert f"{base_name}_bucket" in sample, (
            f"Histogram bucket lines missing for {base_name!r}. "
            f"Did generate_metrics_sample() forget to call "
            f"observe_world_model_observe_step_seconds(...)?"
        )
        assert f"{base_name}_sum" in sample, f"Histogram sum line missing for {base_name!r}"
        assert f"{base_name}_count" in sample, f"Histogram count line missing for {base_name!r}"


# ---------------------------------------------------------------------------
# Forward-compat placeholders — uncomment as C1 / C2 PRs land
# ---------------------------------------------------------------------------
# When C1 lands, uncomment the C1 constants above and replace the
# ``pytest.skip`` below with the same shape as
# ``test_b2_world_model_observe_step_histogram_in_generate_sample``.
# Same for C2.


def test_c1_cloud_metrics_placeholder() -> None:
    """Placeholder for the C1 cloud OTA metric families.

    Stays as ``pytest.skip`` until the C1 PR lands. When C1 ships:

    1. Uncomment ``_C1_COUNTER_FAMILIES`` + ``_C1_HISTOGRAM_FAMILIES`` above.
    2. Replace this body with the same shape as
       ``test_b2_world_model_observe_step_histogram_in_generate_sample``.
    3. The C1 PR's CHANGELOG entry MUST mention this test extension so
       future readers see the dashboard E2E surface grew with the loop.
    """
    pytest.skip("C1 cloud OTA metric families not yet shipped — extend when C1 PR lands")


def test_c2_mission_safety_metrics_placeholder() -> None:
    """Placeholder for the C2 mission + safety projection metric families.

    Stays as ``pytest.skip`` until the C2 PR lands. See
    :func:`test_c1_cloud_metrics_placeholder` for the extension procedure.
    """
    pytest.skip("C2 mission/safety metric families not yet shipped — extend when C2 PR lands")
