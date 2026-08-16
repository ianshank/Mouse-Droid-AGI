"""Regression pins for the code-hygiene-sprint CI wiring (PR #178).

Binary-free assertions over the gate wiring that sprint introduced, in the
same style as ``test_secret_scan_gate.py``: these tests never run the jobs —
what this repo owns is the wiring, and un-pinned wiring is exactly how the
smoke tier silently ran in zero CI paths for months.

Pinned contracts:

* the ``test`` job runs the smoke tier and filters ``hardware``-marked tests;
* the ``performance`` job stays advisory with the shared-runner-calibrated
  instrumentation-overhead budget;
* the ``local-gates`` job keeps running the deterministic scripts/ci.sh-only
  gates in GitHub CI, including the hardcoded-value gate (PR-only, full
  fetch depth — it needs a resolvable base ref);
* the ``security`` job is honestly advisory (``continue-on-error``, no shell
  ``||`` swallow) and skips only the editable local package;
* every ``continue-on-error`` job in ci.yml has an
  ``.github/advisory_stages.yaml`` entry (mirrors
  ``scripts/check_advisory_promotions.py`` as a PR-time signal);
* scripts/ci.sh runs the smoke stage OUTSIDE the ``MOUSEDROID_CI_SLIM`` skip;
* pytest ``addopts`` keeps ``--import-mode=importlib`` (duplicate test
  basenames make prepend mode fragile).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from tests._pyproject import load_pyproject

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_YML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_CI_SH = _REPO_ROOT / "scripts" / "ci.sh"
_ADVISORY_STAGES = _REPO_ROOT / ".github" / "advisory_stages.yaml"


def _load_ci_jobs() -> dict:
    """Parse ci.yml and return its jobs mapping (see test_secret_scan_gate)."""
    data = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "ci.yml did not parse to a mapping"
    jobs = data.get("jobs")
    assert isinstance(jobs, dict), "ci.yml has no jobs mapping"
    return jobs


def _job_run_text(job: dict) -> str:
    """Concatenate every run: block of a job for substring pins."""
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


class TestTestJobTiers:
    """The test job runs smoke and filters hardware-marked tests."""

    def test_smoke_tier_runs_in_test_job(self) -> None:
        run_text = _job_run_text(_load_ci_jobs()["test"])
        assert "pytest tests/smoke" in run_text, (
            "the smoke tier ran in ZERO CI paths before PR #178 — keep it in "
            "the test job (it is sub-10-seconds by construction)"
        )

    def test_coverage_step_filters_hardware_and_uses_importlib(self) -> None:
        run_text = _job_run_text(_load_ci_jobs()["test"])
        assert '-m "not hardware"' in run_text, (
            "hardware-marked tests open real GPIO/serial devices and must not "
            "collect on shared runners (PR #160 rationale, mirrored from ci.sh)"
        )
        assert "--import-mode=importlib" in run_text


class TestPerformanceJob:
    """Advisory performance tier with the runner-calibrated overhead budget."""

    def test_job_is_advisory(self) -> None:
        job = _load_ci_jobs()["performance"]
        assert job.get("continue-on-error") is True, (
            "performance stays advisory: Jetson-calibrated latency budgets "
            "measured 1.17x-1.26x from shared-runner contention alone"
        )

    def test_overhead_budget_env_is_set(self) -> None:
        job = _load_ci_jobs()["performance"]
        envs = [step.get("env", {}) for step in job.get("steps", [])]
        budgets = [
            e["MOUSEDROID_INSTRUMENTATION_OVERHEAD_BUDGET"]
            for e in envs
            if "MOUSEDROID_INSTRUMENTATION_OVERHEAD_BUDGET" in e
        ]
        assert budgets, (
            "the performance job must set MOUSEDROID_INSTRUMENTATION_OVERHEAD_BUDGET "
            "or the 1.15x dedicated-hardware budget leaves the job perma-red "
            "on shared runners"
        )
        assert all(float(b) >= 1.15 for b in budgets)


class TestLocalGatesJob:
    """The deterministic scripts/ci.sh-only gates keep running in GitHub CI."""

    def test_all_seven_gates_present(self) -> None:
        run_text = _job_run_text(_load_ci_jobs()["local-gates"])
        for needle in (
            "check_settings_identity.py",
            "mypy tools/claude_hooks",
            "validate_skill_commands.py",
            "doc_hygiene.py",
            "--cov=tools/claude_hooks",
            "check_no_hardcoded_values.py",
            "check_subsystem_boundaries.py",
        ):
            assert needle in run_text, f"local-gates job lost the {needle} gate"

    def test_hardcoded_value_gate_is_pull_request_only_with_full_history(self) -> None:
        """The gate needs a git diff base — PR-only trigger + full fetch supply one.

        Regression guard for the exact gap the gate's own hard-fail exists to
        catch: on a push event there is no PR base ref to diff against, so
        the step must stay conditioned on pull_request (never run unguarded),
        and the job's checkout must fetch full history so the base branch is
        locally resolvable.
        """
        job = _load_ci_jobs()["local-gates"]
        checkout_step = next(s for s in job["steps"] if "checkout" in str(s.get("uses", "")))
        assert checkout_step.get("with", {}).get("fetch-depth") == 0, (
            "local-gates must fetch full history or the hardcoded-value gate's "
            "base-ref resolution silently loses coverage on shallow checkouts"
        )
        gate_step = next(
            s for s in job["steps"] if "check_no_hardcoded_values.py" in str(s.get("run", ""))
        )
        assert gate_step.get("if") == "github.event_name == 'pull_request'", (
            "without this guard, a push event (no PR base ref) hits the "
            "script's own CI=true/no-base-ref exit(2) and the job goes red "
            "for a reason unrelated to any actual hardcoded value"
        )


class TestSecurityJob:
    """pip-audit is honestly advisory and audits the real dependency tree."""

    def test_job_is_advisory_without_shell_swallow(self) -> None:
        job = _load_ci_jobs()["security"]
        assert job.get("continue-on-error") is True, (
            "advisory-ness must be continue-on-error (visible to "
            "check_advisory_promotions.py), never a shell `||` swallow"
        )
        audit_runs = [
            str(step.get("run", ""))
            for step in job["steps"]
            if "pip-audit" in str(step.get("run", "")) and "install" not in str(step.get("run", ""))
        ]
        assert audit_runs, "security job no longer runs pip-audit"
        for run in audit_runs:
            assert "||" not in run, (
                "a `||` swallow made this job invisible to the advisory "
                "tracker for its whole life — do not reintroduce it"
            )
            assert "--skip-editable" in run, (
                "the editable-installed local package is not on PyPI; without "
                "--skip-editable the audit fails on it instead of real findings"
            )
            assert "--strict" not in run, (
                "--strict escalates the --skip-editable skip itself to a "
                "fatal error ('distribution marked as editable')"
            )


class TestAdvisoryTracking:
    """Every continue-on-error job is tracked in advisory_stages.yaml."""

    def test_all_advisory_jobs_tracked(self) -> None:
        jobs = _load_ci_jobs()
        advisory_jobs = {name for name, job in jobs.items() if job.get("continue-on-error") is True}
        stages = yaml.safe_load(_ADVISORY_STAGES.read_text(encoding="utf-8"))
        tracked = {
            entry["job"] for entry in stages.get("stages", []) if entry.get("workflow") == "ci.yml"
        }
        untracked = advisory_jobs - tracked
        assert not untracked, (
            "continue-on-error jobs without an advisory_stages.yaml entry "
            f"have no promotion clock: {sorted(untracked)}"
        )


class TestCiShSmokeStage:
    """ci.sh runs smoke outside the MOUSEDROID_CI_SLIM skip."""

    def test_smoke_stage_precedes_slim_gate(self) -> None:
        text = _CI_SH.read_text(encoding="utf-8")
        smoke_at = text.find("pytest tests/smoke")
        assert smoke_at != -1, "ci.sh lost its smoke stage"
        slim_at = text.find("MOUSEDROID_CI_SLIM:-0")
        assert slim_at != -1, "ci.sh lost the MOUSEDROID_CI_SLIM gate"
        assert smoke_at < slim_at, (
            "the smoke stage must run BEFORE (outside) the SLIM-gated block — "
            "it is cheap enough for memory-constrained hosts"
        )


def test_pytest_addopts_keeps_importlib_mode() -> None:
    pytest_cfg = load_pyproject()["tool"]["pytest"]["ini_options"]  # type: ignore[index]
    assert "--import-mode=importlib" in pytest_cfg.get("addopts", ""), (
        "duplicate test basenames (e.g. tests/unit/test_profiler.py vs "
        "tests/unit/efficiency/test_profiler.py) make the default prepend "
        "import mode fragile — keep importlib in addopts"
    )
