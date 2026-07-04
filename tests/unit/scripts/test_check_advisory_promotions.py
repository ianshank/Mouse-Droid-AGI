"""Unit tests for the advisory promotion-lag checker (F-020, WS-8.3)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from tests._script_loader import load_script_module

_checker = load_script_module("check_advisory_promotions")

_TODAY = date(2026, 7, 3)


def _workflow(tmp_path: Path, *, advisory_job: str | None) -> Path:
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(exist_ok=True)
    jobs = "  blocking-job:\n    runs-on: ubuntu-latest\n"
    if advisory_job:
        jobs += f"  {advisory_job}:\n    runs-on: ubuntu-latest\n    continue-on-error: true\n"
    (wf_dir / "ci.yml").write_text(f"name: ci\non: push\njobs:\n{jobs}", encoding="utf-8")
    return wf_dir


def _metadata(tmp_path: Path, *, job: str, since: str, window: int) -> Path:
    meta = tmp_path / "advisory_stages.yaml"
    meta.write_text(
        f"stages:\n  - job: {job}\n    workflow: ci.yml\n"
        f"    since: {since}\n    promote_after_days: {window}\n",
        encoding="utf-8",
    )
    return meta


class TestEvaluate:
    def test_tracked_fresh_stage_is_clean(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job="scanner")
        meta = _metadata(tmp_path, job="scanner", since="2026-06-30", window=30)
        warnings = _checker.evaluate(
            _checker.find_advisory_jobs(wf),
            _checker.load_tracked_stages(meta),
            today=_TODAY,
        )
        assert warnings == []

    def test_overdue_promotion_warns(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job="scanner")
        meta = _metadata(tmp_path, job="scanner", since="2026-01-01", window=30)
        warnings = _checker.evaluate(
            _checker.find_advisory_jobs(wf),
            _checker.load_tracked_stages(meta),
            today=_TODAY,
        )
        assert len(warnings) == 1
        assert "promotion overdue" in warnings[0]

    def test_untracked_advisory_job_warns(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job="mystery-stage")
        warnings = _checker.evaluate(
            _checker.find_advisory_jobs(wf),
            [],
            today=_TODAY,
        )
        assert len(warnings) == 1
        assert "untracked advisory stage" in warnings[0]
        assert "mystery-stage" in warnings[0]

    def test_stale_metadata_warns(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job=None)
        meta = _metadata(tmp_path, job="promoted-job", since="2026-06-01", window=30)
        warnings = _checker.evaluate(
            _checker.find_advisory_jobs(wf),
            _checker.load_tracked_stages(meta),
            today=_TODAY,
        )
        assert len(warnings) == 1
        assert "stale metadata" in warnings[0]


class TestCli:
    def test_advisory_exit_zero_despite_warnings(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job="scanner")
        rc = _checker.main(
            ["--workflows-dir", str(wf), "--metadata", str(tmp_path / "absent.yaml")]
        )
        assert rc == 0

    def test_strict_exit_one_on_warnings(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job="scanner")
        rc = _checker.main(
            [
                "--workflows-dir",
                str(wf),
                "--metadata",
                str(tmp_path / "absent.yaml"),
                "--strict",
            ]
        )
        assert rc == 1


class TestRealRepoState:
    def test_repo_advisory_stages_are_all_tracked_and_in_window(self) -> None:
        """The real repo must be warning-free with today pinned to the landing date."""
        repo_root = Path(__file__).resolve().parents[3]
        warnings = _checker.evaluate(
            _checker.find_advisory_jobs(repo_root / ".github" / "workflows"),
            _checker.load_tracked_stages(repo_root / ".github" / "advisory_stages.yaml"),
            today=_TODAY,
        )
        assert warnings == [], warnings


class TestMalformedMetadata:
    """Hand-edited YAML must degrade to a warning, never a traceback."""

    def test_garbled_since_warns_instead_of_crashing(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job="scanner")
        meta = tmp_path / "advisory_stages.yaml"
        meta.write_text(
            "stages:\n  - job: scanner\n    since: not-a-date\n    promote_after_days: 30\n",
            encoding="utf-8",
        )
        warnings = _checker.evaluate(
            _checker.find_advisory_jobs(wf),
            _checker.load_tracked_stages(meta),
            today=_TODAY,
        )
        assert len(warnings) == 1
        assert "malformed metadata for job 'scanner'" in warnings[0]

    def test_missing_window_warns_instead_of_crashing(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job="scanner")
        meta = tmp_path / "advisory_stages.yaml"
        meta.write_text(
            "stages:\n  - job: scanner\n    since: 2026-06-30\n",
            encoding="utf-8",
        )
        warnings = _checker.evaluate(
            _checker.find_advisory_jobs(wf),
            _checker.load_tracked_stages(meta),
            today=_TODAY,
        )
        assert len(warnings) == 1
        assert "malformed metadata" in warnings[0]


def test_yaml_extension_workflows_are_scanned(tmp_path: Path) -> None:
    """GitHub honors .yaml too - an advisory job there must not escape the guard."""
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    (wf_dir / "extra.yaml").write_text(
        "name: extra\non: push\njobs:\n  sneaky:\n    runs-on: ubuntu-latest\n"
        "    continue-on-error: true\n",
        encoding="utf-8",
    )
    assert _checker.find_advisory_jobs(wf_dir) == {"sneaky": "extra.yaml"}


class TestMalformedMetadataFile:
    """Copilot review: a malformed FILE (not just entry) must also degrade."""

    def test_list_root_degrades_to_untracked_warnings(self, tmp_path: Path) -> None:
        wf = _workflow(tmp_path, advisory_job="scanner")
        meta = tmp_path / "advisory_stages.yaml"
        meta.write_text("- job: scanner\n", encoding="utf-8")  # list root, no 'stages' map
        stages = _checker.load_tracked_stages(meta)
        assert stages == []
        warnings = _checker.evaluate(_checker.find_advisory_jobs(wf), stages, today=_TODAY)
        assert len(warnings) == 1
        assert "untracked advisory stage" in warnings[0]

    def test_string_stage_entries_are_filtered(self, tmp_path: Path) -> None:
        meta = tmp_path / "advisory_stages.yaml"
        meta.write_text(
            "stages:\n  - just-a-string\n  - job: ok\n    since: 2026-06-30\n"
            "    promote_after_days: 30\n",
            encoding="utf-8",
        )
        stages = _checker.load_tracked_stages(meta)
        assert [s.get("job") for s in stages] == ["ok"]

    def test_unparseable_yaml_degrades_to_empty(self, tmp_path: Path) -> None:
        meta = tmp_path / "advisory_stages.yaml"
        meta.write_text("stages: [unclosed\n", encoding="utf-8")
        assert _checker.load_tracked_stages(meta) == []
