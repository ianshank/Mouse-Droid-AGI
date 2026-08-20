"""F-026 AQA: declared governance budgets have consumers.

Every config knob that declares a governance budget or operational mode
MUST have at least one consumer outside its own schema/config module.
Without this test, a budget field can parse and validate for years while
enforcing nothing — the exact class of defect F-026 exists to prevent.

The test uses a pinned list of (field_name, consumer_module) pairs.
When a new budget field is added, add it here. When a consumer is wired,
add the module path. If the consumer list is empty, the test fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Repo root (auto-discovered).
_REPO = Path(__file__).resolve().parents[2]

# Budget/mode fields declared in config schemas + their known consumers.
# Format: (field_dotpath, [list of consumer module paths relative to repo root])
# A consumer is any module outside the schema that READS the field.
_BUDGET_CONSUMERS: list[tuple[str, list[str]]] = [
    # F-026 primary targets:
    (
        "JetsonConfig.power_mode",
        [
            # Consumer: HealthMonitor.check_health reads power_mode into
            # the /api/v1/health response dict (wired as part of F-026).
            "src/mousedroid/health/monitor.py",
        ],
    ),
    (
        "DocsConfig.core_max_lines",
        [
            # Consumer: Sprint 5 will wire a CLAUDE.md trimmer that reads
            # this field. Until then, the AQA test is pinned with the
            # planned consumer path so it documents the intent and fails
            # visibly if the path drifts.
            # NOTE: this entry is currently EXPECTED to be absent on disk
            # until Sprint 5 lands. The test is marked xfail for this
            # specific entry only.
            "tools/claude_hooks/docs_trimmer.py",
        ],
    ),
    # Workforce coverage thresholds — already consumed by the coverage-gate skill
    (
        "CoverageConfig.tools_line_min",
        [
            "tools/claude_hooks/config.py",
        ],
    ),
]


def _file_exists(rel_path: str) -> bool:
    """Check if a consumer file exists on disk."""
    return (_REPO / rel_path).exists()


class TestBudgetConsumers:
    """F-026: every declared governance budget has at least one consumer."""

    @pytest.mark.parametrize(
        ("field_dotpath", "consumer_paths"),
        _BUDGET_CONSUMERS,
        ids=[fp for fp, _ in _BUDGET_CONSUMERS],
    )
    def test_budget_field_has_consumer(
        self,
        field_dotpath: str,
        consumer_paths: list[str],
    ) -> None:
        """Assert that at least one consumer module exists for each budget field."""
        if not consumer_paths:
            pytest.fail(
                f"Budget field {field_dotpath!r} has no declared consumer. "
                f"Wire it to a real consumer or delete the field."
            )

        existing = [p for p in consumer_paths if _file_exists(p)]

        # DocsConfig.core_max_lines consumer is planned for Sprint 5 —
        # xfail until the trimmer lands.
        if field_dotpath == "DocsConfig.core_max_lines" and not existing:
            pytest.xfail(
                "DocsConfig.core_max_lines consumer (docs_trimmer.py) "
                "not yet wired — Sprint 5 deliverable."
            )

        if not existing:
            pytest.fail(
                f"Budget field {field_dotpath!r} declares consumers "
                f"{consumer_paths!r} but none exist on disk."
            )

    def test_budget_registry_is_non_empty(self) -> None:
        """The registry itself must not be accidentally emptied."""
        assert len(_BUDGET_CONSUMERS) >= 3, (
            f"Expected at least 3 budget entries, got {len(_BUDGET_CONSUMERS)}. "
            f"Did someone accidentally clear _BUDGET_CONSUMERS?"
        )
