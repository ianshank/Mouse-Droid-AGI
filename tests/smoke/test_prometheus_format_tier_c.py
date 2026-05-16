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
2. **Tier C1 cloud OTA families** (wired by this PR — 4 families:
   downloads counter, sha256 mismatches counter, swaps counter labeled
   by engine_type, download_seconds histogram). Exercised both via
   ``generate_metrics_sample()`` so the scrape endpoint sees seeded
   series from the first poll AND via a dedicated registry to assert
   the per-label rendering.
3. **Tier C2 mission + safety families** — placeholder until the C2 PR
   lands (see ``test_c2_mission_safety_metrics_placeholder``).

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

import re

import pytest

from mousedroid.config.schema import MetricsConfig
from mousedroid.telemetry.metrics import MetricsRegistry, generate_metrics_sample

pytestmark = pytest.mark.smoke

# ---------------------------------------------------------------------------
# Metric name derivation from config (NOT hardcoded)
# ---------------------------------------------------------------------------
_NS = MetricsConfig().namespace  # default: "mousedroid"

# Tier B2 — world-model observe_step latency histogram. Wired in Tier C3.1.
_B2_HISTOGRAM_FAMILIES: tuple[str, ...] = (f"{_NS}_world_model_observe_step_seconds",)

# Tier C1 — cloud OTA weight-update families. Wired by this PR.
_C1_COUNTER_FAMILIES: tuple[str, ...] = (
    f"{_NS}_cloud_weight_update_downloads",
    f"{_NS}_cloud_weight_update_sha256_mismatches",
    f"{_NS}_cloud_weight_update_swaps",
)
_C1_HISTOGRAM_FAMILIES: tuple[str, ...] = (f"{_NS}_cloud_weight_update_download_seconds",)

_HELP_RE = re.compile(r"^# HELP (\S+) .+$", re.MULTILINE)
_TYPE_RE = re.compile(r"^# TYPE (\S+) (counter|gauge|histogram|summary|untyped)$", re.MULTILINE)


def _build_registry_with_c1() -> MetricsRegistry:
    """Build a registry and exercise every Tier C1 family once.

    Used by the per-family render/help/type assertions below. Distinct from
    the ``generate_metrics_sample()`` exercise because that path also seeds
    Tier A / Tier B / Tier C2-future families — keeping a focused builder
    here makes the per-label render asserts narrowly scoped.
    """
    registry = MetricsRegistry(MetricsConfig())
    registry.inc_cloud_weight_update_download("ianshank/mousedroid-policy-v2")
    registry.inc_cloud_weight_update_sha256_mismatch("ianshank/mousedroid-policy-v2")
    registry.observe_cloud_weight_update_download_seconds(2.5)
    registry.inc_cloud_weight_update_swap("policy")
    registry.inc_cloud_weight_update_swap("world_model")
    return registry


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


def test_c1_cloud_metrics_render_in_prometheus_output() -> None:
    """All four Tier C1 families surface on render after the first observation."""
    registry = _build_registry_with_c1()
    text = registry.render_prometheus()

    for family in _C1_COUNTER_FAMILIES:
        total = f"{family}_total"
        assert total in text, f"Tier C1 counter missing from /metrics: {total}"

    for family in _C1_HISTOGRAM_FAMILIES:
        assert f"{family}_bucket" in text, f"Tier C1 histogram missing _bucket: {family}"
        assert f"{family}_sum" in text, f"Tier C1 histogram missing _sum: {family}"
        assert f"{family}_count" in text, f"Tier C1 histogram missing _count: {family}"


def test_c1_cloud_metrics_seeded_in_generate_metrics_sample() -> None:
    """``generate_metrics_sample()`` seeds every C1 family for promtool / CI.

    Without this seeding, the first scrape after a Jetson restart would
    omit the C1 families entirely (Prometheus exposition format only
    emits series the registry has observed at least once). That would
    silently break the Grafana cloud-OTA panel + the
    ``WeightUpdateSHA256Mismatch`` alert rule (no series to evaluate
    against).
    """
    sample = generate_metrics_sample()
    for family in _C1_COUNTER_FAMILIES:
        total = f"{family}_total"
        assert total in sample, f"generate_metrics_sample() missing C1 counter: {total}"
    for family in _C1_HISTOGRAM_FAMILIES:
        assert (
            f"{family}_bucket" in sample
        ), f"generate_metrics_sample() missing C1 histogram: {family}"


def test_c1_help_and_type_paired() -> None:
    """Every C1 family has both ``# HELP`` and ``# TYPE`` declarations.

    Prometheus accepts series without ``# HELP`` but downstream tooling
    (promtool, alerting rule linters) emits warnings — pinning the pair
    keeps the exposition output strict.
    """
    registry = _build_registry_with_c1()
    text = registry.render_prometheus()
    help_names = set(_HELP_RE.findall(text))
    type_names = {name for name, _ in _TYPE_RE.findall(text)}
    for family in _C1_COUNTER_FAMILIES:
        total = f"{family}_total"
        assert total in help_names, f"missing HELP for {total}"
        assert total in type_names, f"missing TYPE for {total}"
    for family in _C1_HISTOGRAM_FAMILIES:
        assert family in help_names, f"missing HELP for {family}"
        assert family in type_names, f"missing TYPE for {family}"


def test_c1_counter_labels_include_repo_id_and_engine_type() -> None:
    """Labeled counter sample lines emit the expected label dimensions.

    Pins the label contract so Grafana panels using ``sum by (repo_id)``
    or ``sum by (engine_type)`` aggregations stay stable across refactors.
    """
    registry = _build_registry_with_c1()
    text = registry.render_prometheus()
    assert (
        f'{_NS}_cloud_weight_update_downloads_total{{repo_id="ianshank/mousedroid-policy-v2"}}'
        in text
    )
    assert (
        f"{_NS}_cloud_weight_update_sha256_mismatches_total"
        f'{{repo_id="ianshank/mousedroid-policy-v2"}}' in text
    )
    assert f'{_NS}_cloud_weight_update_swaps_total{{engine_type="policy"}}' in text
    assert f'{_NS}_cloud_weight_update_swaps_total{{engine_type="world_model"}}' in text


# ---------------------------------------------------------------------------
# Forward-compat placeholder — uncomment as C2 PR lands
# ---------------------------------------------------------------------------
# When C2 lands, replace the ``pytest.skip`` below with the same shape as
# ``test_c1_cloud_metrics_seeded_in_generate_metrics_sample``.


def test_c2_mission_safety_metrics_placeholder() -> None:
    """Placeholder for the C2 mission + safety projection metric families.

    Stays as ``pytest.skip`` until the C2 PR lands. When C2 ships:

    1. Add the C2 metric-name constants alongside ``_C1_COUNTER_FAMILIES``.
    2. Replace this body with the same shape as
       ``test_c1_cloud_metrics_seeded_in_generate_metrics_sample``.
    3. The C2 PR's CHANGELOG entry MUST mention this test extension so
       future readers see the dashboard E2E surface grew with the loop.
    """
    pytest.skip("C2 mission/safety metric families not yet shipped — extend when C2 PR lands")
