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
* the functional / user-journey / security tiers run in the blocking ``test``
  job and in ci.sh, and never in the advisory ``security`` job (F-028);
* EVERY discovered tests/<tier>/ reaches a CI path or carries a documented
  exemption -- generic, so the next orphaned tier cannot slip through;
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


def _ci_sh_commands() -> str:
    """scripts/ci.sh with comment lines stripped.

    Asserting a tier name against the raw file text is satisfiable by a
    *comment* -- including the explanatory comments this repo writes above each
    stage. Deleting the pytest line while keeping the comment would leave the
    pin green, which is precisely the un-pinned-wiring failure this module
    exists to prevent.
    """
    return "\n".join(
        line
        for line in _CI_SH.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


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

    def test_all_eight_gates_present(self) -> None:
        run_text = _job_run_text(_load_ci_jobs()["local-gates"])
        for needle in (
            "check_settings_identity.py",
            "mypy tools/claude_hooks",
            "validate_skill_commands.py",
            "doc_hygiene.py",
            "tools.ratchet_budgets",
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


_ORPHAN_TIERS = ("tests/functional", "tests/user_journey", "tests/security")

# Tiers deliberately absent from hosted CI, each with the reason. Same posture
# as _ALLOWED_CROSS_SUBSYSTEM_IMPORTS in scripts/check_subsystem_boundaries.py:
# a ratchet, not a bypass valve. Adding an entry is a reviewable policy
# decision; it is not a way to silence the gate.
_CI_EXEMPT_TIERS: dict[str, str] = {
    "hardware": (
        "rover-only: opens real GPIO / serial / CSI devices, so it must not "
        "collect on shared runners. Runs on the self-hosted Jetson via "
        "scripts/jetson_full_validation.sh; .github/workflows/harness.yml "
        "documents the deliberate omission."
    ),
}


def _discover_test_tiers() -> set[str]:
    """Every tests/<tier>/ directory that actually holds tests.

    Discovered rather than listed, so a tier added tomorrow is covered without
    editing this file. That is the whole point: F-028 fixed three orphaned
    tiers, but a hardcoded roster would let the *next* orphan slip through
    exactly the same way.
    """
    tests_root = _REPO_ROOT / "tests"
    return {
        d.name
        for d in tests_root.iterdir()
        if d.is_dir() and d.name != "__pycache__" and any(d.rglob("test_*.py"))
    }


class TestEveryTierReachesCi:
    """No test tier may run in zero CI paths -- generically, not by roster."""

    def test_every_tier_is_wired_or_explicitly_exempt(self) -> None:
        ci_sh = _ci_sh_commands()
        # yaml.safe_load drops comments, so the joined run: text is already
        # comment-free -- unlike the raw ci.yml source.
        ci_yml = "\n".join(_job_run_text(job) for job in _load_ci_jobs().values())
        orphans = sorted(
            tier
            for tier in _discover_test_tiers()
            if tier not in _CI_EXEMPT_TIERS
            and f"tests/{tier}" not in ci_sh
            and f"tests/{tier}" not in ci_yml
        )
        assert not orphans, (
            f"test tier(s) {orphans} run in ZERO CI paths. Either wire them into "
            "scripts/ci.sh and the blocking `test` job, or add an entry to "
            "_CI_EXEMPT_TIERS with a documented reason. A tier nobody runs rots "
            "invisibly -- that is what F-028 existed to fix."
        )

    def test_exemptions_are_not_stale(self) -> None:
        """An exemption for a tier that no longer exists is dead policy."""
        discovered = _discover_test_tiers()
        stale = sorted(t for t in _CI_EXEMPT_TIERS if t not in discovered)
        assert not stale, (
            f"_CI_EXEMPT_TIERS names tier(s) {stale} that no longer exist -- "
            "drop the entry rather than leaving a rule nobody can trip"
        )

    def test_every_exemption_carries_a_reason(self) -> None:
        """A bare exemption is indistinguishable from an oversight."""
        for tier, reason in _CI_EXEMPT_TIERS.items():
            assert reason.strip(), f"exemption for {tier!r} has no documented reason"


class TestOrphanTierWiring:
    """F-028: the functional / user-journey / security tiers reach a CI path.

    All three ran in ZERO CI paths -- absent from ci.sh and ci.yml -- so they
    could rot invisibly, exactly as the smoke tier did before PR #178.

    Scope note: this closes a *wiring* gap, not a coverage hole. The
    ``RegexInjectionFilter`` unit coverage in
    ``tests/unit/security/test_injection_filter.py`` (11 tests) already ran in
    the coverage-gated ``test`` job; ``tests/security/`` adds the pre-egress
    path through the gateway seam on top of it.
    """

    def test_all_three_tiers_run_in_the_test_job(self) -> None:
        run_text = _job_run_text(_load_ci_jobs()["test"])
        for tier in _ORPHAN_TIERS:
            assert tier in run_text, (
                f"{tier} must run in the blocking `test` job — it previously "
                "ran in no CI path at all (F-028)"
            )

    def test_tiers_are_not_in_the_advisory_security_job(self) -> None:
        """The `security` job is continue-on-error and would swallow failures."""
        security = _load_ci_jobs()["security"]
        assert security.get("continue-on-error") is True, (
            "precondition changed: the security job is no longer advisory, so "
            "this guard needs rethinking rather than deleting"
        )
        run_text = _job_run_text(security)
        assert "tests/security" not in run_text, (
            "tests/security must NOT run in the advisory `security` job — "
            "continue-on-error would swallow every failure and recreate the "
            "orphan-tier problem wearing a disguise"
        )

    def test_ci_sh_runs_all_three_tiers(self) -> None:
        commands = _ci_sh_commands()
        for tier in _ORPHAN_TIERS:
            assert tier in commands, (
                f"ci.sh lost the {tier} stage (F-028) — note this checks the "
                "comment-stripped command text, so an explanatory comment "
                "naming the tier does not satisfy it"
            )

    def test_orphan_tier_stage_precedes_slim_gate(self) -> None:
        text = _ci_sh_commands()
        stage_at = text.find("pytest tests/functional")
        assert stage_at != -1, "ci.sh lost its functional/user-journey/security stage"
        slim_at = text.find("MOUSEDROID_CI_SLIM:-0")
        assert slim_at != -1, "ci.sh lost the MOUSEDROID_CI_SLIM gate"
        assert stage_at < slim_at, (
            "the three tiers run in ~2.5s total — keep them OUTSIDE the "
            "SLIM-gated block, same rationale as the smoke stage"
        )


def test_pytest_addopts_keeps_importlib_mode() -> None:
    pytest_cfg = load_pyproject()["tool"]["pytest"]["ini_options"]  # type: ignore[index]
    assert "--import-mode=importlib" in pytest_cfg.get("addopts", ""), (
        "duplicate test basenames (e.g. tests/unit/test_profiler.py vs "
        "tests/unit/efficiency/test_profiler.py) make the default prepend "
        "import mode fragile — keep importlib in addopts"
    )
