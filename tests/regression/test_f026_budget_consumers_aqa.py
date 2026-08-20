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
            # Consumer: tools/claude_hooks/docs_trimmer.py checks that root
            # CLAUDE.md does not exceed core_max_lines (F-026 / F-024 Phase 6).
            "tools/claude_hooks/docs_trimmer.py",
        ],
    ),
    # Workforce coverage threshold — consumed by the dedicated tools coverage
    # gate in scripts/ci.sh (reads load_config().coverage.tools_line_min).
    # NOTE: the schema module (tools/claude_hooks/config.py) does NOT count
    # as a consumer — declaring a field is not reading it.
    (
        "CoverageConfig.tools_line_min",
        [
            "scripts/ci.sh",
        ],
    ),
]


def _file_exists(rel_path: str) -> bool:
    """Check if a consumer file exists on disk."""
    return (_REPO / rel_path).exists()


def _reads_field(rel_path: str, field_name: str) -> bool:
    """Check that a consumer file's text actually references the budget field."""
    path = _REPO / rel_path
    if not path.is_file():
        return False
    return field_name in path.read_text(encoding="utf-8", errors="replace")


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
        """Assert at least one consumer exists AND references each budget field."""
        if not consumer_paths:
            pytest.fail(
                f"Budget field {field_dotpath!r} has no declared consumer. "
                f"Wire it to a real consumer or delete the field."
            )

        existing = [p for p in consumer_paths if _file_exists(p)]

        if not existing:
            pytest.fail(
                f"Budget field {field_dotpath!r} declares consumers "
                f"{consumer_paths!r} but none exist on disk."
            )

        # Existence is not consumption: at least one consumer must actually
        # reference the field by name, so a consumer that silently drops the
        # read (or is hollowed out) fails this gate.
        field_name = field_dotpath.rsplit(".", 1)[-1]
        reading = [p for p in existing if _reads_field(p, field_name)]

        if not reading:
            pytest.fail(
                f"Budget field {field_dotpath!r} declares consumers "
                f"{existing!r} but none of them reference {field_name!r}. "
                f"The field is declared yet enforced nowhere — the exact "
                f"defect class F-026 exists to prevent."
            )

    def test_budget_registry_is_non_empty(self) -> None:
        """The registry itself must not be accidentally emptied."""
        assert len(_BUDGET_CONSUMERS) >= 3, (
            f"Expected at least 3 budget entries, got {len(_BUDGET_CONSUMERS)}. "
            f"Did someone accidentally clear _BUDGET_CONSUMERS?"
        )
