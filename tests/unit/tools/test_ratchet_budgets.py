"""Unit tests for the ratchet-budget early-warning checker.

Covers the pure counting/check helpers and the CLI contract: advisory mode
always exits 0 (warnings are printed, never fatal); ``--strict`` flips
warnings into exit 1; a disabled config skips checking entirely.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.claude_hooks.config import RatchetBudgetItem
from tools.ratchet_budgets import (
    check_all_budgets,
    check_budget_item,
    count_marker_occurrences,
    main,
)


def _write_source(tmp_path: Path, rel_path: str, *, occurrences: int, marker: str = "noqa") -> None:
    target = tmp_path / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"x = 1  # {marker}\n" for _ in range(occurrences)), encoding="utf-8")


def _item(**overrides: object) -> RatchetBudgetItem:
    defaults: dict[str, object] = {
        "name": "noqa",
        "marker": "noqa",
        "scope_glob": "src/mousedroid/**/*.py",
        "ceiling": 5,
        "warn_threshold": 3,
    }
    defaults.update(overrides)
    return RatchetBudgetItem(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# count_marker_occurrences
# ---------------------------------------------------------------------------


class TestCountMarkerOccurrences:
    def test_counts_across_multiple_files(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=3)
        _write_source(repo, "src/mousedroid/pkg/b.py", occurrences=2)
        assert count_marker_occurrences(repo, "src/mousedroid/**/*.py", "noqa") == 5

    def test_ignores_files_outside_scope_glob(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=1)
        _write_source(repo, "tests/b.py", occurrences=9)
        assert count_marker_occurrences(repo, "src/mousedroid/**/*.py", "noqa") == 1

    def test_counts_multiple_occurrences_per_line(self, repo: Path) -> None:
        target = repo / "src" / "mousedroid" / "c.py"
        target.parent.mkdir(parents=True)
        target.write_text("x = 1  # noqa noqa\n", encoding="utf-8")
        assert count_marker_occurrences(repo, "src/mousedroid/**/*.py", "noqa") == 2

    def test_empty_scope_yields_zero(self, repo: Path) -> None:
        assert count_marker_occurrences(repo, "src/mousedroid/**/*.py", "noqa") == 0


# ---------------------------------------------------------------------------
# check_budget_item
# ---------------------------------------------------------------------------


class TestCheckBudgetItem:
    def test_healthy_count_yields_no_warnings(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=2)
        assert check_budget_item(repo, _item()) == []

    def test_over_warn_threshold_but_under_ceiling_warns(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=4)
        warnings = check_budget_item(repo, _item())
        assert len(warnings) == 1
        assert "early-warning" in warnings[0]

    def test_over_ceiling_warns_with_exceeds_message(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=6)
        warnings = check_budget_item(repo, _item())
        assert len(warnings) == 1
        assert "exceeds" in warnings[0]

    def test_over_ceiling_reports_only_one_warning_not_two(self, repo: Path) -> None:
        # A ceiling breach implies the warn threshold is crossed too — must
        # not double-report the same drift.
        _write_source(repo, "src/mousedroid/a.py", occurrences=6)
        assert len(check_budget_item(repo, _item())) == 1

    def test_no_warn_threshold_only_ceiling_gates(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=4)
        assert check_budget_item(repo, _item(warn_threshold=None)) == []

    def test_exactly_at_ceiling_warns_but_does_not_exceed(self, repo: Path) -> None:
        # ceiling is inclusive (count > ceiling triggers, not count >= ceiling),
        # so sitting exactly on it still crosses warn_threshold and reports the
        # approaching-budget message, not the harder "exceeds" one.
        _write_source(repo, "src/mousedroid/a.py", occurrences=5)
        warnings = check_budget_item(repo, _item())
        assert len(warnings) == 1
        assert "early-warning" in warnings[0]

    def test_exactly_at_warn_threshold_is_healthy(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=3)
        assert check_budget_item(repo, _item()) == []


# ---------------------------------------------------------------------------
# check_all_budgets
# ---------------------------------------------------------------------------


class TestCheckAllBudgets:
    def test_aggregates_across_items(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=6, marker="noqa")
        _write_source(repo, "src/mousedroid/b.py", occurrences=1, marker="type: ignore")
        items = [
            _item(name="noqa", marker="noqa"),
            _item(name="type_ignore", marker="type: ignore"),
        ]
        warnings = check_all_budgets(repo, items)
        assert len(warnings) == 1
        assert warnings[0].startswith("noqa:")

    def test_empty_items_yields_no_warnings(self, repo: Path) -> None:
        assert check_all_budgets(repo, []) == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_advisory_mode_exits_zero_despite_warnings(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=25)
        rc = main(["--repo-root", str(repo)])
        assert rc == 0
        assert "WARN:" in capsys.readouterr().out

    def test_strict_mode_exits_one_on_warnings(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=25)
        assert main(["--repo-root", str(repo), "--strict"]) == 1

    def test_strict_mode_exits_zero_when_clean(self, repo: Path) -> None:
        assert main(["--repo-root", str(repo), "--strict"]) == 0

    def test_clean_run_prints_ok_message(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--repo-root", str(repo)]) == 0
        assert "all tracked budgets within range" in capsys.readouterr().out

    def test_disabled_config_skips_checking_entirely(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=999)
        (repo / ".claude").mkdir()
        (repo / ".claude" / "workforce.yaml").write_text(
            "ratchet_budgets:\n    enabled: false\n", encoding="utf-8"
        )
        rc = main(["--repo-root", str(repo), "--strict"])
        assert rc == 0
        assert "all tracked budgets within range" in capsys.readouterr().out

    def test_custom_budget_from_config_is_honored(self, repo: Path) -> None:
        _write_source(repo, "src/mousedroid/a.py", occurrences=2, marker="# custom-marker")
        (repo / ".claude").mkdir()
        (repo / ".claude" / "workforce.yaml").write_text(
            "ratchet_budgets:\n"
            "    items:\n"
            "        - name: custom\n"
            "          marker: '# custom-marker'\n"
            "          ceiling: 1\n",
            encoding="utf-8",
        )
        assert main(["--repo-root", str(repo), "--strict"]) == 1

    def test_broken_config_degrades_to_advisory_warn_not_a_crash(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """load_config() raising ConfigError must not crash the CLI.

        The module docstring promises "always exits 0 unless --strict" — an
        invalid workforce.yaml (unknown key, extra="forbid") is exactly the
        kind of environment problem this advisory tool must survive, same as
        any other finding.
        """
        (repo / ".claude").mkdir()
        (repo / ".claude" / "workforce.yaml").write_text(
            "ratchet_budgets:\n    unknown_key: 1\n", encoding="utf-8"
        )
        rc = main(["--repo-root", str(repo)])
        assert rc == 0
        assert "WARN: could not load workforce config" in capsys.readouterr().out

    def test_broken_config_under_strict_exits_one(self, repo: Path) -> None:
        (repo / ".claude").mkdir()
        (repo / ".claude" / "workforce.yaml").write_text(
            "ratchet_budgets:\n    unknown_key: 1\n", encoding="utf-8"
        )
        assert main(["--repo-root", str(repo), "--strict"]) == 1
