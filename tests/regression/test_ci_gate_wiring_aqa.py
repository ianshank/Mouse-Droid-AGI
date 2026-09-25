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
* the ``security`` job is blocking (promoted 2026-09-16) with no shell ``||``
  swallow, and skips only the editable local package;
* every ``continue-on-error`` job in ci.yml has an
  ``.github/advisory_stages.yaml`` entry (mirrors
  ``scripts/check_advisory_promotions.py`` as a PR-time signal);
* scripts/ci.sh runs the smoke stage OUTSIDE the ``MOUSEDROID_CI_SLIM`` skip;
* the functional / user-journey / security tiers run in the blocking ``test``
  job and in ci.sh, and never in the ``security`` (pip-audit) job (F-028);
* those three tiers use the SAME pytest marker expression in all three places
  that run them (ci.yml, ci.sh, scripts/validations/F-028.sh), so local, CI,
  and the feature's own validation command cannot silently run different
  test sets;
* EVERY discovered tests/<tier>/ reaches a CI path or carries a documented
  exemption -- generic, so the next orphaned tier cannot slip through;
* no live doc claims ``tests/security/`` is the ONLY coverage of the
  pre-egress injection filter -- it is not, and F-028's own proposal and
  peer-review both said so, the latter marking it CONFIRMED;
* pytest ``addopts`` keeps ``--import-mode=importlib`` (duplicate test
  basenames make prepend mode fragile);
* the ``typecheck`` job installs both the ``telemetry`` and ``mlflow`` extras,
  and no other job runs mypy. ``mypy --strict`` is sensitive to the whole
  installed dependency set in BOTH directions — an absent optional lib degrades
  to ``Any`` and trips ``untyped-decorator``, while a present one that ships real
  types can turn a required ``cast`` into ``redundant-cast`` — so a second mypy
  invocation under different extras reports *different* errors, not more of them.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from tests._pyproject import load_pyproject

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_YML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_CI_SH = _REPO_ROOT / "scripts" / "ci.sh"
_ADVISORY_STAGES = _REPO_ROOT / ".github" / "advisory_stages.yaml"
_F028_VALIDATION = _REPO_ROOT / "scripts" / "validations" / "F-028.sh"


def _load_ci_jobs() -> dict:
    """Parse ci.yml and return its jobs mapping (see test_secret_scan_gate)."""
    data = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "ci.yml did not parse to a mapping"
    jobs = data.get("jobs")
    assert isinstance(jobs, dict), "ci.yml has no jobs mapping"
    return jobs


def _strip_comments(text: str) -> str:
    """Drop whole-line ``#`` comments, keeping ``#`` inside a command.

    Asserting a tier name against raw text is satisfiable by a *comment* --
    including the explanatory comments this repo writes above each stage.
    Deleting the pytest line while keeping the comment would leave the pin
    green, which is precisely the un-pinned-wiring failure this module exists
    to prevent.

    This applies to ``ci.yml`` as much as to the shell scripts: ``yaml``
    parses a ``run:`` block as a **literal scalar**, so every ``#`` line inside
    it survives ``safe_load`` verbatim. An earlier revision asserted the
    opposite in a comment and left the ci.yml-side pins comment-satisfiable --
    the same defect, fixed on one side only.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _shell_commands(path: Path) -> str:
    """A shell script's text with comment lines stripped."""
    return _strip_comments(path.read_text(encoding="utf-8"))


def _ci_sh_commands() -> str:
    """scripts/ci.sh, comment-stripped (see :func:`_strip_comments`)."""
    return _shell_commands(_CI_SH)


def _logical_lines(text: str) -> list[str]:
    """Join backslash-continued shell lines into one logical command each.

    The orphan-tier invocation spans four physical lines everywhere it appears,
    so a per-line regex would never see the command and its ``-m`` expression
    together.

    Only an ODD number of trailing backslashes continues a line: ``foo\\``
    ends a command with a literal backslash. Treating that as a continuation
    would splice an unrelated command onto the front of the next one, and the
    marker regex takes the first match on the joined line.
    """
    commands: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        trailing_backslashes = len(line) - len(line.rstrip("\\"))
        if trailing_backslashes % 2 == 1:
            buffer += line[:-1].strip() + " "
            continue
        commands.append((buffer + line.strip()).strip())
        buffer = ""
    if buffer:
        commands.append(buffer.strip())
    return commands


_ORPHAN_TIER_INVOCATION = "pytest tests/functional"
# ``-m`` followed by a *quoted* argument. ``python -m pytest`` is unquoted, so
# it cannot be mistaken for the marker expression.
_MARKER_EXPR = re.compile(r'-m\s+"([^"]*)"')


def _orphan_tier_markers(text: str, *, site: str) -> list[str]:
    """Every marker expression guarding an orphan-tier run in this text.

    A list, not the first match: one file can invoke the tiers more than once,
    and returning only the first would make *intra*-site drift invisible --
    the same instance-not-class mistake this module keeps correcting.

    Empty when the text does not run the tiers. Raises when it runs them with
    no ``-m`` filter, since that would collect hardware-marked tests.
    """
    markers: list[str] = []
    for command in _logical_lines(text):
        if _ORPHAN_TIER_INVOCATION not in command:
            continue
        match = _MARKER_EXPR.search(command)
        assert match is not None, (
            f"{site} runs the orphan tiers with no -m marker expression, so "
            f"hardware-marked tests would collect there: {command}"
        )
        markers.append(match.group(1))
    return markers


# Sites that must be found, whatever else discovery turns up. Their absence is
# a wiring regression, not a naming change, so it fails loudly rather than
# shrinking the comparison set to whatever happens to still match.
_REQUIRED_MARKER_SITES = frozenset(
    {".github/workflows/ci.yml", "scripts/ci.sh", "scripts/validations/F-028.sh"}
)


def _discover_marker_sites() -> dict[str, list[str]]:
    """Every place in the repo that runs the orphan tiers, and its markers.

    Discovered rather than listed. A hardcoded roster of three was exactly the
    bug: `make behaviour` is a fourth site, and it agreed only by luck. A fifth
    added tomorrow is covered here without editing this file.

    Values are lists so a file invoking the tiers twice contributes both
    markers -- ci.yml in particular is collected across *every* job, not just
    the first one found.
    """
    sites: dict[str, list[str]] = {}

    ci_yml_markers: list[str] = []
    for job_name, job in _load_ci_jobs().items():
        ci_yml_markers.extend(
            _orphan_tier_markers(_job_run_text(job), site=f"ci.yml job {job_name}")
        )
    if ci_yml_markers:
        sites[".github/workflows/ci.yml"] = ci_yml_markers

    candidates = [_CI_SH, _REPO_ROOT / "Makefile"]
    candidates.extend(sorted((_REPO_ROOT / "scripts" / "validations").glob("*.sh")))
    for path in candidates:
        if not path.is_file():
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        markers = _orphan_tier_markers(_shell_commands(path), site=rel)
        if markers:
            sites[rel] = markers
    return sites


def _job_run_text(job: dict) -> str:
    """Concatenate every run: block of a job, comment-stripped.

    Comment-stripped because ``yaml`` keeps ``#`` lines inside a ``run:``
    block scalar verbatim, so a substring pin over the raw text is satisfiable
    by a commented-out invocation -- and the negative pins below would go
    falsely red on a comment that merely *names* a tier.
    """
    return _strip_comments("\n".join(str(step.get("run", "")) for step in job.get("steps", [])))


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

    def test_all_nine_gates_present(self) -> None:
        run_text = _job_run_text(_load_ci_jobs()["local-gates"])
        for needle in (
            "check_settings_identity.py",
            "mypy tools/claude_hooks",
            "validate_skill_commands.py",
            "doc_hygiene.py",
            "tools.claude_hooks.docs_trimmer",
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
    """pip-audit is blocking and audits the real dependency tree.

    Promoted advisory -> blocking on 2026-09-16 (tech-debt Wave 1), inside its
    60-day window. The half of this class that mattered is unchanged: the job
    must never hide findings behind a shell ``||`` swallow, which is how it spent
    its entire life before being un-swallowed, and it must keep the exact
    ``--skip-editable``/no-``--strict`` flag pair that makes its output mean
    anything. Only the advisory-ness assertion inverted.
    """

    def test_job_is_blocking_without_shell_swallow(self) -> None:
        job = _load_ci_jobs()["security"]
        assert job.get("continue-on-error") is not True, (
            "security is blocking since 2026-09-16 — pip-audit reported zero "
            "vulnerabilities across the resolved [dev,telemetry,mcp] set. "
            "Re-adding continue-on-error also needs an advisory_stages.yaml "
            "entry, or check_advisory_promotions.py WARNs 'untracked advisory "
            "stage'."
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


class TestTypecheckJobDependencySet:
    """``mypy --strict`` runs against the extras its result depends on.

    ``mypy --strict`` is sensitive to the *whole* installed dependency set, in
    both directions, and this job is the only place that is checked:

    * **Absent** optional libs degrade to ``Any`` under
      ``--ignore-missing-imports``, and strict mode then flags the annotations
      that reference them. Without the ``telemetry`` extra (aiohttp), the three
      ``@web.middleware`` functions in ``telemetry/auth.py`` and
      ``telemetry/server/_lifecycle.py`` report ``untyped-decorator``.
    * **Present** optional libs that ship real types can make a previously
      necessary construct redundant. With ``mlflow`` installed, a ``cast`` that
      ``no-any-return`` required when mlflow was absent becomes
      ``redundant-cast``.

    Both were real failures. The second was invisible until a mypy step was
    added to the advisory ``mlflow-extras`` job — which then hit the FIRST one,
    because that job installs no ``telemetry`` extra. The resolution was to run
    one mypy invocation against one superset dependency list, here, in a blocking
    matrixed job. This class is the pin that stops the two diverging again;
    nothing asserted the extras before, which is precisely how it happened.
    """

    @staticmethod
    def _typecheck_install_extras() -> str:
        """Return only the ``pip install -e`` extras, with comments stripped.

        Comment lines are excluded deliberately, and this is not hypothetical:
        the first revision of this class matched the raw ``run:`` block, whose
        prose explains at length *why* telemetry and mlflow are installed. It
        found "mlflow" in the comment and passed even with the extra deleted from
        the install line — a pin that could not fail. See :func:`_strip_comments`
        above, which documents the same trap for the tier assertions and exists
        for exactly this reason; this helper is its narrower cousin, keeping only
        real ``pip install -e`` commands rather than all non-comment lines.
        """
        job = _load_ci_jobs()["typecheck"]
        commands = [
            line.strip()
            for step in job["steps"]
            for line in str(step.get("run", "")).splitlines()
            if line.strip().startswith("pip install -e")
        ]
        assert commands, "typecheck job has no `pip install -e` line to inspect"
        return "\n".join(commands)

    def test_typecheck_installs_telemetry_extra(self) -> None:
        """aiohttp must be present or the middleware decorators read as untyped."""
        assert "telemetry" in self._typecheck_install_extras(), (
            "typecheck must install the telemetry extra — without aiohttp, "
            "mypy --strict reports untyped-decorator on the @web.middleware "
            "functions in telemetry/auth.py and telemetry/server/_lifecycle.py"
        )

    def test_typecheck_installs_mlflow_extra(self) -> None:
        """mlflow must be present or its redundant-cast class is unreachable."""
        assert "mlflow" in self._typecheck_install_extras(), (
            "typecheck must install the mlflow extra — mlflow-skinny ships real "
            "type information, so errors that only appear when it is present "
            "(redundant-cast on MlflowClient returns) are otherwise invisible "
            "to every CI job"
        )

    def test_typecheck_actually_runs_mypy_strict(self) -> None:
        run_text = _job_run_text(_load_ci_jobs()["typecheck"])
        assert "mypy" in run_text, "typecheck job no longer invokes mypy"
        assert "--strict" in run_text, "typecheck job runs mypy without --strict"

    def test_mlflow_extras_does_not_run_mypy(self) -> None:
        """Keep exactly one mypy invocation, under one dependency set.

        A second invocation under a narrower extras combination reports
        *different* errors rather than more of them, which is a false signal, not
        extra coverage.
        """
        assert "mypy" not in _job_run_text(_load_ci_jobs()["mlflow-extras"]), (
            "mlflow-extras must not run mypy: it installs no telemetry extra, so "
            "mypy --strict fails there on untyped-decorator regardless of mlflow. "
            "The mlflow typecheck belongs in the `typecheck` job, which installs "
            "the superset."
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


#: Regex for the deterministic gate invocations in scripts/ci.sh. Three forms:
#: a module run as ``-m pkg.mod``; a script run as ``tools/x.py`` /
#: ``scripts/x.py``; and an inline ``-c "from pkg.mod import ..."``, which is how
#: the workforce-config validation runs and which an earlier version of this
#: regex missed entirely (found by peer review).
_CI_SH_GATE_RE = re.compile(
    r"(?:-m\s+(?P<module>[a-z_][a-z0-9_.]*)"
    r'|"?\$PYTHON_BIN"?\s+(?P<script>(?:tools|scripts)/[\w/]+\.py)'
    r'|-c\s+"?(?:from|import)\s+(?P<inline>[a-z_][a-z0-9_.]*))'
)

#: Generic tooling that is a *runner*, not a gate — matching these would assert
#: that "pytest appears in a workflow", which proves nothing about coverage.
_CI_SH_RUNNERS = frozenset({"pytest", "ruff", "mypy", "pip", "coverage"})

#: ci.sh gates deliberately absent from hosted CI, each with the reason. Same
#: ratchet posture as _CI_EXEMPT_TIERS: an entry is a reviewable decision.
_CI_EXEMPT_GATES: dict[str, str] = {
    "mousedroid.cli.validate_pillars": (
        "redundant with pytest, not unwired: ci.sh runs it as a ~50ms "
        "importability smoke check, and tests/unit/cli/test_validate_pillars_cli.py "
        "invokes the same main(['--dry-run']) inside the blocking `test` job. A "
        "workflow step would re-run what pytest already covers."
    ),
    "scripts/check_branch_coverage.py": (
        "local-only by design, and ci.yml's own Stage 3c banner says so: it "
        "re-runs the whole unit/property/integration suite to compute a "
        "changed-file branch delta, which the blocking `test` job already pays "
        "for once at the line level. Duplicating it would roughly double the "
        "job's wall-clock for a second view of the same run."
    ),
    "mousedroid.main": (
        "not a gate: ci.sh invokes the application entrypoint itself for a "
        "startup smoke check, which the e2e and smoke tiers cover in the "
        "blocking `test` job."
    ),
    "scripts/generate_package_map.py": (
        "covered by tests/regression/test_f053_package_map_aqa.py in the "
        "blocking `test` job (regenerate-and-diff with CRLF/path normalisation). "
        "ci.sh --check is the fast local mirror; a duplicate workflow step would "
        "only re-assert what pytest already gates, and editing ci.yml requires "
        "the workflow OAuth scope this account's token does not carry."
    ),
}


def _discover_ci_sh_gates() -> set[str]:
    """Every deterministic gate scripts/ci.sh invokes *in one of three forms*.

    Discovered by regex rather than listed, for the same reason
    :func:`_discover_test_tiers` is: a hardcoded roster would let the *next*
    ci.sh-only gate slip through exactly as ``tools.claude_hooks.docs_trimmer``
    did — it guarded the root CLAUDE.md line budget and no workflow ran it, so a
    PR that blew the budget passed all 17 jobs.

    **Scope, stated because the first version of this docstring overstated it.**
    This covers ``-m pkg.mod``, ``$PYTHON_BIN path/to/x.py``, and inline
    ``-c "from pkg.mod import ..."``. A gate added as a heredoc, as
    ``bash scripts/foo.sh``, or as a direct binary is still invisible, so this
    narrows the window rather than closing it. Widen the regex when a fourth form
    appears; do not read the sweep as exhaustive.
    """
    text = _CI_SH.read_text(encoding="utf-8")
    gates: set[str] = set()
    for match in _CI_SH_GATE_RE.finditer(text):
        for group in ("module", "script", "inline"):
            value = match.group(group)
            if not value:
                continue
            if group == "script":
                gates.add(value)
            elif value not in _CI_SH_RUNNERS:
                # An inline `-c "from pkg.mod import ..."` is credited to the
                # module it imports: `tools.claude_hooks.config` is the gate.
                gates.add(value)
    return gates


def _all_workflow_text() -> str:
    """The text of every workflow, with YAML comments stripped.

    Text rather than parsed structure: a gate may be invoked from any workflow
    (``validate.py`` lives in harness.yml, not ci.yml), and this only ever asks
    whether a gate is mentioned *somewhere*, which the enumerated per-job pins
    above then sharpen.

    Comments are stripped because counting them let a gate named only in prose
    satisfy the sweep — delete the ``run:`` line, keep a comment mentioning the
    module, and the gate is unwired while this test stays green. The new
    ``docs_trimmer`` step ships exactly such a comment, so it is not theoretical.
    """
    workflows = _REPO_ROOT / ".github" / "workflows"
    lines: list[str] = []
    for path in sorted(workflows.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            lines.append(line.split(" #", 1)[0] if " #" in line else line)
    return "\n".join(lines)


class TestEveryCiShGateReachesCi:
    """No deterministic ci.sh gate may run in zero GitHub Actions paths.

    ``scripts/ci.sh`` is the authoritative local superset, so a gate added there
    feels wired — but only a workflow enforces it on a PR. This sweep is the
    generic guard; the enumerated per-job pins above are the specific ones.
    """

    def test_every_gate_is_wired_or_explicitly_exempt(self) -> None:
        workflow_text = _all_workflow_text()
        orphans = sorted(
            gate
            for gate in _discover_ci_sh_gates()
            if gate not in _CI_EXEMPT_GATES and gate not in workflow_text
        )
        assert not orphans, (
            f"scripts/ci.sh gate(s) {orphans} run in ZERO GitHub Actions paths, so a "
            "PR that breaks them passes every job. Either add a workflow step, or "
            "add an entry to _CI_EXEMPT_GATES with a documented reason."
        )

    def test_the_sweep_actually_finds_gates(self) -> None:
        """Guard against a regex that silently matches nothing.

        Without this, a typo in ``_CI_SH_GATE_RE`` would turn the sweep above into
        an unconditional pass — the exact failure mode it exists to prevent.
        """
        discovered = _discover_ci_sh_gates()
        assert len(discovered) >= 10, f"the ci.sh gate sweep found only {discovered}"
        # One per recognised form, so losing a branch of the regex fails here
        # rather than silently shrinking what the sweep can see.
        assert "scripts/check_subsystem_boundaries.py" in discovered
        assert "tools.claude_hooks.docs_trimmer" in discovered
        assert "tools.claude_hooks.config" in discovered, (
            "the inline -c form is not being discovered — ci.sh validates the "
            "workforce config that way, and it was invisible until peer review"
        )

    def test_exemptions_are_not_stale(self) -> None:
        """An exemption for a gate ci.sh no longer runs is dead policy."""
        discovered = _discover_ci_sh_gates()
        stale = sorted(gate for gate in _CI_EXEMPT_GATES if gate not in discovered)
        assert not stale, (
            f"_CI_EXEMPT_GATES names gate(s) {stale} that scripts/ci.sh no longer "
            "invokes — drop the entry rather than leaving a rule nobody can trip"
        )

    def test_every_exemption_carries_a_reason(self) -> None:
        for gate, reason in _CI_EXEMPT_GATES.items():
            assert reason.strip(), f"exemption for {gate!r} has no documented reason"


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

    def test_tiers_are_not_in_the_pip_audit_security_job(self) -> None:
        """``tests/security`` belongs to the ``test`` job, not the audit job.

        Rethought (not deleted) when ``security`` was promoted advisory ->
        blocking on 2026-09-16 — this guard's own predecessor asked for exactly
        that. The original reason was that ``continue-on-error`` would swallow
        every failure in the tier, recreating the F-028 orphan-tier problem
        wearing a disguise. That specific hazard is gone, but the separation
        still holds for a second, independent reason: the ``security`` job audits
        the *dependency tree* (``pip-audit``) and installs a different extras set
        than the tier needs, so co-locating them would couple a Python test tier
        to a supply-chain scan's lifecycle. The tier's home is the ``test`` job,
        asserted by ``test_all_three_tiers_run_in_the_test_job`` above; this is
        the negative half of that pair.
        """
        security = _load_ci_jobs()["security"]
        run_text = _job_run_text(security)
        assert "tests/security" not in run_text, (
            "tests/security must NOT run in the `security` (pip-audit) job — "
            "it runs in the blocking `test` job (F-028). The audit job exists "
            "to scan dependencies, not to host a test tier."
        )
        assert "pip-audit" in run_text, (
            "precondition: this guard assumes `security` is the pip-audit job"
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


class TestOrphanTierMarkerParity:
    """Every site that runs the orphan tiers uses the SAME marker expression.

    Four run them today -- CI (``ci.yml``), the local superset (``ci.sh``),
    F-028's own ``validation_command`` (``scripts/validations/F-028.sh``), and
    the ``make behaviour`` target. They drifted the moment the first was fixed
    in isolation: ci.yml said ``not hardware and not slow`` while ci.sh and
    F-028.sh said ``not hardware``, so a green local run and a green validation
    command asserted a *different test set* than CI gated on.

    Latent while no ``slow`` marks live in those tiers -- and a trap the moment
    one does. Sites are **discovered**, not listed: an earlier cut named the
    three from the review finding and missed ``make behaviour``, which agreed
    only by luck. Naming them would repeat the mistake this class exists to
    correct -- fixing the instances, not the class.
    """

    def test_the_known_wiring_sites_are_all_discovered(self) -> None:
        """Discovery must not silently shrink to whatever still matches.

        If `ci.sh` stops running the tiers, the parity assertion below would
        pass vacuously over the survivors -- measured: reverting ci.sh and
        ci.yml to the pre-F-028 base leaves parity and the hardware filter
        both green over the two remaining sites, and only this assertion red.
        Pin the floor separately.
        """
        missing = sorted(_REQUIRED_MARKER_SITES - set(_discover_marker_sites()))
        assert not missing, f"these no longer run the orphan tiers at all: {missing}"

    def test_marker_expression_is_identical_across_every_site(self) -> None:
        markers = _discover_marker_sites()
        distinct = {m for site_markers in markers.values() for m in site_markers}
        assert len(distinct) == 1, (
            "the orphan tiers run under different pytest marker expressions "
            f"depending on who invokes them: {markers}. Local runs and F-028's "
            "validation command must gate on exactly what CI gates on, or "
            "'it passes locally' stops meaning anything."
        )

    def test_marker_still_excludes_hardware_marked_tests(self) -> None:
        """Parity alone is satisfiable by deleting the filter everywhere.

        Pin the safety property too: ``hardware``-marked tests open real
        GPIO / serial / CSI devices and must never collect on a shared runner.
        """
        for site, site_markers in _discover_marker_sites().items():
            for marker in site_markers:
                assert "not hardware" in marker, (
                    f"{site} no longer excludes hardware-marked tests from the "
                    f"orphan tiers (marker={marker!r})"
                )


_INJECTION_FILTER_UNIT_TESTS = (
    _REPO_ROOT / "tests" / "unit" / "security" / "test_injection_filter.py"
)

# Whitespace-normalised so a claim wrapped across lines is still caught. A
# line-oriented grep missed the copy in proposal.md for exactly that reason,
# which is how a sweep that reported "corrected everywhere" left two behind.
_ONLY_COVERAGE_CLAIM = re.compile(
    r"only\s+(?:known\s+)?cover(?:age|s)?[^.]{0,120}?"
    r"(?:pre-egress|injection|RegexInjectionFilter)"
    r"|(?:pre-egress|injection|RegexInjectionFilter)[^.]{0,120}?only\s+cover",
    re.IGNORECASE,
)

# What redeems a mention of the claim. A doc may state it, quote it, or refute
# it -- it may not leave it bare, because bare is how it reads as current
# truth. Naming the unit-test file, saying "unit coverage", or marking the
# claim REFUTED / false all discharge it.
#
# Framed this way rather than as a phrase blacklist with an exemption list:
# a verdict table SHOULD be able to quote the claim it refutes, and a proposal
# SHOULD be able to state the narrow version ("only coverage through the
# gateway seam"). Both are fine; neither is fine unqualified.
_CLAIM_QUALIFIERS = (
    "test_injection_filter",
    "unit coverage",
    "refuted",
    "was false",
)
# Normalised characters after the match in which a qualifier must appear.
_QUALIFIER_WINDOW = 400


def _tracked_docs() -> list[Path]:
    """Every prose surface a future engineer might read as current truth.

    Sourced from ``git ls-files`` rather than a directory-prefix glob list.
    The prior version (``*.md``, ``docs/**/*.md``, ``openspec/**/*.md``,
    ``.claude/**/*.md``, ``src/**/*.md``) missed 15 tracked docs outright --
    including ``tests/agent.md``, an agent-facing instructions file this same
    doc-reconciliation sprint corrected a stale coverage-floor claim in, so
    the gap was not hypothetical. A hardcoded directory roster is exactly the
    pattern ``TestOrphanTierWiring`` below exists to replace with discovery
    for CI tiers; the doc inventory had the identical shape of gap.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", "*.md"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    relpaths = sorted(line for line in result.stdout.splitlines() if line.strip())
    return [_REPO_ROOT / relpath for relpath in relpaths]


class TestOrphanTierNarrativeAccuracy:
    """No doc may claim tests/security is the ONLY coverage of the filter.

    F-028's own proposal and peer-review said exactly that, and the
    peer-review marked it **CONFIRMED** -- a governance record asserting a
    falsehood as verified. It survived one sweep that reported "corrected
    everywhere", because that sweep fixed the places it remembered rather than
    the places that had it.

    The claim is checkable against the tree, so this pins the doc to the tree
    rather than blacklisting a phrase: while unit coverage of the filter
    exists, no live doc may say the security tier is the only coverage. What
    ``tests/security/`` uniquely exercises is the **gateway seam** -- a wiring
    gap, not a coverage hole, and the difference is the whole justification
    for F-028's scope.
    """

    def test_unit_coverage_of_the_injection_filter_exists(self) -> None:
        """The tree fact the claim contradicts. If this ever stops being true,
        the sibling assertion below is measuring nothing."""
        assert _INJECTION_FILTER_UNIT_TESTS.is_file(), (
            f"{_INJECTION_FILTER_UNIT_TESTS} is gone -- either restore it or "
            "revisit the narrative pinned below, which depends on it existing"
        )
        body = _INJECTION_FILTER_UNIT_TESTS.read_text(encoding="utf-8")
        assert body.count("def test_") >= 2, (
            "the injection filter's unit coverage is what makes 'the security "
            "tier is the only coverage' false; it must actually hold tests"
        )

    def test_no_live_doc_states_the_claim_unqualified(self) -> None:
        offenders: list[str] = []
        for path in _tracked_docs():
            normalised = " ".join(path.read_text(encoding="utf-8").split())
            for match in _ONLY_COVERAGE_CLAIM.finditer(normalised):
                window = normalised[match.start() : match.end() + _QUALIFIER_WINDOW].lower()
                if not any(q in window for q in _CLAIM_QUALIFIERS):
                    offenders.append(
                        f"{path.relative_to(_REPO_ROOT).as_posix()}: "
                        f"...{normalised[match.start() : match.end()]}..."
                    )
        assert not offenders, (
            "these state, unqualified, that tests/security is the only coverage "
            "of the pre-egress injection filter:\n  " + "\n  ".join(offenders) + "\n"
            "It is not: tests/unit/security/test_injection_filter.py already ran "
            "in the coverage-gated `test` job. Either narrow the claim to the "
            "gateway seam (true, and what F-028 actually fixed) or mark it "
            "refuted -- but do not leave it reading as current truth."
        )


def test_pytest_addopts_keeps_importlib_mode() -> None:
    pytest_cfg = load_pyproject()["tool"]["pytest"]["ini_options"]  # type: ignore[index]
    assert "--import-mode=importlib" in pytest_cfg.get("addopts", ""), (
        "duplicate test basenames (e.g. tests/unit/test_profiler.py vs "
        "tests/unit/efficiency/test_profiler.py) make the default prepend "
        "import mode fragile — keep importlib in addopts"
    )
