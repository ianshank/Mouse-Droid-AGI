"""Unit tests for ``config/prometheus/alerts.yml`` (F-019, WS-5).

Repo-wide rule hygiene (every rule carries a severity and a ``config_ref``
annotation pointing back at the schema field or documented default), and --
for EVERY group, not just one -- every ``mousedroid_*`` identifier an
expression references must resolve against ``generate_metrics_sample()``: the
same known-names contract the Grafana dashboard test enforces.

That check used to cover only the F-019 LLM-gateway group, so two MCP alerts
(one of them severity ``page``) shipped against ``mousedroid_mcp_tool_calls``, a
series nothing emits, and could never fire (F-052 task 6.1). ``promtool check
rules`` in CI passed them throughout: it validates syntax, not whether a rule
can ever be true. Whether the rules FIRE is proven separately, by
``config/prometheus/alerts_test.yml`` under ``promtool test rules``.
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

    def test_every_expr_references_only_rendered_metrics(
        self, known_metric_names: set[str]
    ) -> None:
        """Every group, every rule: an alert on a series nothing emits is dead.

        promtool cannot catch this -- a rule over a nonexistent series is
        valid PromQL that is simply never true.
        """
        dead = {
            rule["alert"]: sorted(
                set(re.findall(r"mousedroid_[A-Za-z0-9_]+", str(rule["expr"]))) - known_metric_names
            )
            for rule in _all_rules(_load())
        }
        dead = {alert: names for alert, names in dead.items() if names}
        assert not dead, (
            f"alert expressions reference metrics absent from generate_metrics_sample(), "
            f"so they can never fire: {dead}"
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


_ARTIFACT_GROUP = "mousedroid_artifact_integrity"
_PLAYBOOK = _REPO_ROOT / "docs" / "playbooks" / "artifact-integrity-fail.md"


def _rule(alert: str) -> dict[str, Any]:
    return next(r for r in _all_rules(_load()) if r.get("alert") == alert)


class TestArtifactIntegrityGroup:
    """F-052 task 6.1: the refusals both counters' docstrings say should page."""

    def test_group_and_alerts_present_and_paging(self) -> None:
        group = next(g for g in _load()["groups"] if g["name"] == _ARTIFACT_GROUP)
        severities = {r["alert"]: r["labels"]["severity"] for r in group["rules"]}
        assert severities == {
            "ModelArtifactDigestMismatch": "page",
            "CloudWeightUpdateDigestMismatch": "page",
        }

    def test_the_boot_path_rule_checks_the_value_not_the_rate(self) -> None:
        """The plan said "page on any non-zero rate". That alert could never fire.

        Both writers run inside build_orchestrator before the telemetry server
        exists, so the first scrape sees the final count and it never changes:
        rate()/increase() over it is 0 forever. Proven by promtool in
        config/prometheus/alerts_test.yml; pinned here so it cannot be
        "corrected" back to the docstring's wording.
        """
        expr = str(_rule("ModelArtifactDigestMismatch")["expr"])
        assert "rate(" not in expr
        assert "increase(" not in expr
        assert "mousedroid_model_artifact_sha256_mismatches_total" in expr

    def test_the_ota_rule_sees_the_first_refusal(self) -> None:
        """The counter is pure-add, so the first refusal is born at 1.

        increase() alone needs two samples of an existing series and never
        fires on that. The ``unless ... offset`` arm is what catches it.
        """
        expr = " ".join(str(_rule("CloudWeightUpdateDigestMismatch")["expr"]).split())
        assert "increase(mousedroid_cloud_weight_update_sha256_mismatches_total" in expr
        assert "unless mousedroid_cloud_weight_update_sha256_mismatches_total offset" in expr

    def test_both_rules_point_operators_at_a_playbook_that_exists(self) -> None:
        for alert in ("ModelArtifactDigestMismatch", "CloudWeightUpdateDigestMismatch"):
            description = str(_rule(alert)["annotations"]["description"])
            assert "docs/playbooks/artifact-integrity-fail.md" in description, alert
        assert _PLAYBOOK.is_file(), f"{_PLAYBOOK} is referenced but missing"


class TestSafetyViolationSeesTheFirstViolation:
    def test_the_rule_has_the_born_non_zero_arm(self) -> None:
        """A bare ``increase(...[1m]) > 0`` missed every rover's FIRST violation.

        The counter does not exist until the first violation creates it at 1,
        and increase() never sees a series being born. Proven by promtool in
        config/prometheus/alerts_test.yml; the arm that fixes it is pinned here.
        """
        expr = " ".join(str(_rule("SafetyViolation")["expr"]).split())
        assert "increase(mousedroid_safety_violations_total[1m])" in expr
        assert "unless mousedroid_safety_violations_total offset 1m" in expr
        assert _rule("SafetyViolation")["labels"]["severity"] == "critical"


_RULE_TESTS = _REPO_ROOT / "config" / "prometheus" / "alerts_test.yml"
_CI = _REPO_ROOT / ".github" / "workflows" / "ci.yml"


class TestTheRuleUnitTestsAreLive:
    """``alerts_test.yml`` is only evidence while something runs it."""

    def test_ci_runs_the_rule_unit_tests(self) -> None:
        assert "promtool test rules config/prometheus/alerts_test.yml" in _CI.read_text(
            encoding="utf-8"
        )

    def test_the_rule_tests_load_the_shipped_rules(self) -> None:
        data = yaml.safe_load(_RULE_TESTS.read_text(encoding="utf-8"))
        assert data["rule_files"] == ["alerts.yml"]

    def test_every_rule_this_change_touched_is_exercised(self) -> None:
        data = yaml.safe_load(_RULE_TESTS.read_text(encoding="utf-8"))
        exercised = {
            check["alertname"] for case in data["tests"] for check in case["alert_rule_test"]
        }
        assert {
            "SafetyViolation",
            "ModelArtifactDigestMismatch",
            "CloudWeightUpdateDigestMismatch",
            "MCPCircuitOpen",
            "MCPRateLimitedSurge",
        } <= exercised

    def test_the_rule_tests_pass_when_promtool_is_available(self) -> None:
        """Run them locally too, when a promtool binary is at hand.

        CI's ``prometheus-check`` job always runs them; this lets a developer
        with promtool on PATH (or ``MOUSEDROID_PROMTOOL`` set) get the same
        answer from pytest instead of discovering it after the push.
        """
        import os
        import shutil
        import subprocess

        promtool = os.environ.get("MOUSEDROID_PROMTOOL") or shutil.which("promtool")
        if not promtool:
            pytest.skip("promtool not available (CI's prometheus-check job runs these)")
        result = subprocess.run(
            [promtool, "test", "rules", str(_RULE_TESTS)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
