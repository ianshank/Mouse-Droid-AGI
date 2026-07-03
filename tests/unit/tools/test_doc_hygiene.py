"""Unit tests for the advisory doc-hygiene guard (F-016).

Covers the pure ``check_doc`` helper and the CLI contract: advisory mode
always exits 0 (warnings are printed, never fatal); ``--strict`` flips
warnings into exit 1; thresholds come from flags, not scattered literals.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.doc_hygiene import _DEFAULT_MAX_BYTES, _DEFAULT_MAX_DONE_MARKS, check_doc, main


@pytest.fixture
def healthy_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "NEXT_STEPS.md"
    doc.write_text("# Next steps\n\n1. Do the thing.\n", encoding="utf-8")
    return doc


@pytest.fixture
def oversized_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "BLOATED.md"
    doc.write_text("x" * (_DEFAULT_MAX_BYTES + 1), encoding="utf-8")
    return doc


@pytest.fixture
def done_heavy_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "CHANGELOGGY.md"
    doc.write_text("done ✅\n" * (_DEFAULT_MAX_DONE_MARKS + 1), encoding="utf-8")
    return doc


class TestCheckDoc:
    def test_healthy_doc_yields_no_warnings(self, healthy_doc: Path) -> None:
        warnings = check_doc(
            healthy_doc,
            max_bytes=_DEFAULT_MAX_BYTES,
            max_done_marks=_DEFAULT_MAX_DONE_MARKS,
        )
        assert warnings == []

    def test_oversized_doc_warns_on_bytes(self, oversized_doc: Path) -> None:
        warnings = check_doc(
            oversized_doc,
            max_bytes=_DEFAULT_MAX_BYTES,
            max_done_marks=_DEFAULT_MAX_DONE_MARKS,
        )
        assert len(warnings) == 1
        assert "byte" in warnings[0]

    def test_done_heavy_doc_warns_on_marks(self, done_heavy_doc: Path) -> None:
        warnings = check_doc(
            done_heavy_doc,
            max_bytes=_DEFAULT_MAX_BYTES,
            max_done_marks=_DEFAULT_MAX_DONE_MARKS,
        )
        assert len(warnings) == 1
        assert "CHANGELOG.md" in warnings[0]

    def test_missing_file_is_a_warning_not_a_crash(self, tmp_path: Path) -> None:
        warnings = check_doc(
            tmp_path / "absent.md",
            max_bytes=_DEFAULT_MAX_BYTES,
            max_done_marks=_DEFAULT_MAX_DONE_MARKS,
        )
        assert len(warnings) == 1
        assert "unreadable" in warnings[0]

    def test_thresholds_are_parameters_not_baked_in(self, healthy_doc: Path) -> None:
        # A tiny budget must flag even the healthy doc — proves the caller's
        # thresholds are authoritative.
        warnings = check_doc(healthy_doc, max_bytes=1, max_done_marks=0)
        assert warnings


class TestCli:
    def test_advisory_mode_exits_zero_despite_warnings(
        self, oversized_doc: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main([str(oversized_doc)])
        assert rc == 0
        assert "WARN:" in capsys.readouterr().out

    def test_strict_mode_exits_one_on_warnings(self, oversized_doc: Path) -> None:
        assert main([str(oversized_doc), "--strict"]) == 1

    def test_strict_mode_exits_zero_when_clean(self, healthy_doc: Path) -> None:
        assert main([str(healthy_doc), "--strict"]) == 0

    def test_flag_overrides_apply(self, healthy_doc: Path) -> None:
        assert main([str(healthy_doc), "--max-bytes", "1", "--strict"]) == 1
