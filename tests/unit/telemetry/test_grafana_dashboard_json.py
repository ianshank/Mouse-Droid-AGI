"""Structural validation for ``docs/grafana_dashboard.json``.

Locks in invariants that catch silent regressions when panels are added or
edited:

- The JSON parses (no trailing-comma / unquoted-key drift).
- Every panel has a stable ``id`` and a non-empty ``title``.
- Every Prometheus expression referenced by a panel target uses only
  metric names that are emitted by :func:`generate_metrics_sample`. Catches
  the typo-on-rename failure mode where a metric is renamed in
  ``MetricsRegistry`` but its Grafana query keeps the old name and silently
  renders an empty panel forever.
- The PR-B2 panels added for the PR-A2 metric families are present.

The test is intentionally lightweight — no Grafana CLI dependency, no
schema-version coupling — so it runs in the default unit suite without
extra extras.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_DASHBOARD = Path("docs/grafana_dashboard.json")


@pytest.fixture(scope="module")
def dashboard() -> dict[str, object]:
    """Load and parse the Grafana dashboard JSON once per session."""
    if not _DASHBOARD.exists():
        pytest.skip(f"{_DASHBOARD} not present in this checkout")
    return json.loads(_DASHBOARD.read_text(encoding="utf-8"))


class TestDashboardStructure:
    """Schema-level invariants — no Grafana version coupling."""

    def test_dashboard_is_valid_json(self, dashboard: dict[str, object]) -> None:
        # If the fixture loaded successfully, JSON is valid. Assert
        # explicitly so a future maintainer reading the test understands.
        assert isinstance(dashboard, dict)

    def test_dashboard_has_panels(self, dashboard: dict[str, object]) -> None:
        panels = dashboard.get("panels")
        assert isinstance(panels, list), "panels must be a JSON array"
        assert len(panels) > 0, "dashboard must declare at least one panel"

    def test_every_panel_has_id_and_title(self, dashboard: dict[str, object]) -> None:
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        for panel in panels:
            assert isinstance(panel, dict)
            assert "id" in panel, f"panel missing id: {panel!r}"
            assert isinstance(panel["id"], int)
            assert panel.get("title"), f"panel id={panel['id']} has empty title"

    def test_panel_ids_are_unique(self, dashboard: dict[str, object]) -> None:
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        ids = [p["id"] for p in panels if isinstance(p, dict)]
        assert len(ids) == len(set(ids)), f"duplicate panel ids: {ids!r}"


class TestPanelExpressionsReferenceKnownMetrics:
    """Every Prometheus expr in a panel must use a metric the registry emits."""

    @pytest.fixture(scope="class")
    def known_metric_names(self) -> set[str]:
        """All metric names emitted by ``generate_metrics_sample``.

        ``generate_metrics_sample`` is responsible for exercising every metric
        family the project considers operationally relevant — including the
        PR-A2 conditional families (replay/VLA/VLM), which are exercised by
        explicit ``inc_*`` / ``observe_*`` calls inside the sample helper.
        That means this fixture's set is the single source of truth: any
        Grafana query referencing a metric that is *not* in the sample is
        either a typo, a stale rename, or a feature-flag-gated metric that
        must be in ``sample_omits`` below with a documented reason.

        Adding metric names directly to a static allowed-set would mask the
        exact rename-drift this test is designed to catch.
        """
        from mousedroid.telemetry.metrics import generate_metrics_sample

        sample = generate_metrics_sample()
        # Every metric line begins with the metric name. Bucket / sum / count
        # are suffixes of histogram base names. Extract via regex.
        names: set[str] = set()
        for line in sample.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Either ``name value`` or ``name{labels} value``.
            m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", stripped)
            if m:
                names.add(m.group(1))
        return names

    def test_all_panel_exprs_reference_known_metrics(
        self,
        dashboard: dict[str, object],
        known_metric_names: set[str],
    ) -> None:
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        # No exemptions. This used to whitelist the MCP family as "not in the
        # sample, but the rendered names are these" -- and one of those
        # asserted names was wrong: the counter rendered
        # ``mousedroid_mcp_tool_calls_total_total``, so the MCP panel queried
        # a series nothing emitted and this test, by construction, could not
        # notice. generate_metrics_sample() now seeds the MCP family, so every
        # panel expression is checked against what is really rendered. A
        # family that genuinely cannot be seeded should be fixed in the
        # sample, not exempted here by an unverified name (F-052 task 6.1).
        for panel in panels:
            assert isinstance(panel, dict)
            for target in panel.get("targets", []):
                expr = target.get("expr", "")
                if not isinstance(expr, str) or not expr:
                    continue
                # Extract every ``mousedroid_*`` identifier from the expr.
                referenced = set(re.findall(r"mousedroid_[A-Za-z0-9_]+", expr))
                unknown = referenced - known_metric_names
                assert not unknown, (
                    f"panel id={panel['id']} title={panel['title']!r} references "
                    f"unknown metric(s): {sorted(unknown)}. Either rename the "
                    f"query or add the metric to MetricsRegistry."
                )


class TestPrB2PanelsPresent:
    """PR-B2 must add Grafana panels covering each PR-A2 metric family."""

    @pytest.fixture(scope="class")
    def panel_titles(self, dashboard: dict[str, object]) -> list[str]:
        panels = dashboard["panels"]
        assert isinstance(panels, list)
        return [p["title"] for p in panels if isinstance(p, dict)]

    @pytest.mark.parametrize(
        "needle",
        [
            "Replay Records",  # outcome counter panel
            "VLA Inference Latency",  # histogram quantile panel
            "VLA Timeouts",  # timeout-by-mode panel
            "VLM Progress Cache",  # cache hit-rate panel
            # Tier C3.1 — world-model observe_step histogram panel
            "World-Model observe_step Latency",
            # F-019 — LLM-gateway families (PR #115)
            "LLM Token Usage",
            "LLM Gateway Latency",
            "LLM Gateway Served",
            "LLM Latency Budget",
        ],
    )
    def test_panel_covering_metric_family_exists(
        self, panel_titles: list[str], needle: str
    ) -> None:
        """Each PR-A2 / Tier-C metric family must have at least one panel title that
        references it (substring match — title format is operator-tunable)."""
        matches = [t for t in panel_titles if needle.lower() in t.lower()]
        assert matches, f"No panel title contains {needle!r}. Existing titles: {panel_titles}"
