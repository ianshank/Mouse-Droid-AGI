"""Unit tests for ``config/prometheus/alerts.yml`` (F-019, WS-5).

Repo-wide rule hygiene (every rule carries a severity and a ``config_ref``
annotation pointing back at the schema field or documented default) plus the
F-019 LLM-gateway group: its four alerts exist and every ``mousedroid_*``
identifier they reference resolves against ``generate_metrics_sample()`` —
the same known-names contract the Grafana dashboard test enforces.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from mousedroid.telemetry.metrics import generate_metrics_sample

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALERTS = _REPO_ROOT / "config" / "prometheus" / "alerts.yml"

_LLM_GROUP = "mousedroid_llm_gateway"
_LLM_ALERTS = (
    "LLMGatewayLatencyHigh",
    "LLMLatencyBudgetExceededSpike",
    "LLMGatewayDegradedServing",
    "LLMTokenBurnHigh",
)


def _load() -> dict[str, Any]:
    data = yaml.safe_load(_ALERTS.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _all_rules(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [rule for group in data["groups"] for rule in group["rules"]]


@pytest.fixture(scope="module")
def known_metric_names() -> set[str]:
    names: set[str] = set()
    for line in generate_metrics_sample().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", stripped)
        if match:
            names.add(match.group(1))
    return names


class TestRepoWideRuleHygiene:
    def test_parses_and_has_groups(self) -> None:
        assert _load()["groups"], "alerts.yml must declare at least one group"

    def test_every_rule_has_severity_and_config_ref(self) -> None:
        for rule in _all_rules(_load()):
            name = rule.get("alert", "<unnamed>")
            assert rule.get("labels", {}).get("severity"), f"{name}: missing labels.severity"
            assert rule.get("annotations", {}).get("config_ref"), (
                f"{name}: missing annotations.config_ref (thresholds must trace "
                "back to a schema field or a documented operator default)"
            )


class TestLlmGatewayGroup:
    def test_group_and_alert_names_present(self) -> None:
        data = _load()
        groups = {g["name"]: g for g in data["groups"]}
        assert _LLM_GROUP in groups
        declared = {r["alert"] for r in groups[_LLM_GROUP]["rules"]}
        assert declared == set(_LLM_ALERTS)

    def test_llm_exprs_reference_known_metrics(self, known_metric_names: set[str]) -> None:
        data = _load()
        group = next(g for g in data["groups"] if g["name"] == _LLM_GROUP)
        for rule in group["rules"]:
            referenced = set(re.findall(r"mousedroid_[A-Za-z0-9_]+", str(rule["expr"])))
            unknown = referenced - known_metric_names
            assert not unknown, (
                f"{rule['alert']}: expr references metrics absent from "
                f"generate_metrics_sample(): {sorted(unknown)}"
            )

    def test_degraded_serving_watches_the_secondary_tier(self) -> None:
        data = _load()
        group = next(g for g in data["groups"] if g["name"] == _LLM_GROUP)
        rule = next(r for r in group["rules"] if r["alert"] == "LLMGatewayDegradedServing")
        assert 'tier="secondary"' in str(rule["expr"])
