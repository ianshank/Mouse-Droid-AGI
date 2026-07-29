"""Regression pins for the Phase-1 ci.sh OOM guard.

The guard lives in ``scripts/jetson_full_validation.sh`` and its slim-mode
partner lives in ``scripts/ci.sh``. Both are bash — we pin the *source-text
contracts* rather than execute them (the guard only fires meaningfully
inside a Jetson container). This mirrors the existing
``tests/regression/test_host_bootstrap_script.py::TestSourceContract``
pattern.

The invariants pinned here fell out of the 2026-07-12 trunk-sync validation
run: Phase-1 ci.sh SIGKILL'd (rc=137, OOM) with no memory guard or retry;
this test file locks the mitigation in place so a future edit can't silently
regress the contract.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JFV = _REPO_ROOT / "scripts" / "jetson_full_validation.sh"
_CI = _REPO_ROOT / "scripts" / "ci.sh"


def _jfv() -> str:
    return _JFV.read_text(encoding="utf-8")


def _ci() -> str:
    return _CI.read_text(encoding="utf-8")


class TestOomGuardEnvVars:
    """The 3 tunables MUST stay env-overridable with documented defaults."""

    def test_ulimit_kb_default_is_env_overridable(self) -> None:
        # First-attempt vmem cap. Default must expand cleanly under `${VAR:-N}`.
        assert 'PHASE1_CI_ULIMIT_KB="${MOUSEDROID_VALIDATION_PHASE1_CI_ULIMIT_KB:-' in _jfv(), (
            "PHASE1_CI_ULIMIT_KB must be env-overridable with a default"
        )

    def test_retry_ulimit_kb_default_is_env_overridable(self) -> None:
        assert (
            'PHASE1_CI_RETRY_ULIMIT_KB="${MOUSEDROID_VALIDATION_PHASE1_CI_RETRY_ULIMIT_KB:-'
            in _jfv()
        ), "PHASE1_CI_RETRY_ULIMIT_KB must be env-overridable with a default"

    def test_oom_retry_is_env_overridable(self) -> None:
        assert 'PHASE1_CI_OOM_RETRY="${MOUSEDROID_VALIDATION_PHASE1_CI_OOM_RETRY:-' in _jfv(), (
            "PHASE1_CI_OOM_RETRY must be env-overridable (operator kill-switch)"
        )

    def test_env_vars_documented_in_header(self) -> None:
        # The header lists every operator-facing tunable; new ones must join.
        src = _jfv()
        for var in (
            "MOUSEDROID_VALIDATION_PHASE1_CI_ULIMIT_KB",
            "MOUSEDROID_VALIDATION_PHASE1_CI_RETRY_ULIMIT_KB",
            "MOUSEDROID_VALIDATION_PHASE1_CI_OOM_RETRY",
        ):
            # The var appears both in the # header block AND the assignment.
            assert src.count(var) >= 2, f"{var} must be both documented + assigned"


class TestOomGuardWrapper:
    """The bash wrapper function `run_phase1_ci_container` must survive."""

    def test_wrapper_function_exists(self) -> None:
        assert "run_phase1_ci_container()" in _jfv(), "Phase-1 OOM guard wrapper must be defined"

    def test_wrapper_applies_ulimit_before_ci_sh(self) -> None:
        # `ulimit -v ${PHASE1_CI_ULIMIT_KB}` must precede `bash scripts/ci.sh`
        # in the docker exec command chain. Without it, the ulimit is a no-op.
        src = _jfv()
        # Look for the exact pattern: ulimit -v $LIMIT && cd && bash scripts/ci.sh
        assert "ulimit -v ${PHASE1_CI_ULIMIT_KB}" in src
        assert "ulimit -v ${PHASE1_CI_RETRY_ULIMIT_KB}" in src

    def test_wrapper_retries_only_on_137(self) -> None:
        # Retry MUST be gated on rc==137 (SIGKILL). Retrying on rc=1 masks real
        # test failures; retrying on any non-zero would infinite-loop test bugs.
        assert "rc} -eq 137" in _jfv(), "retry must be gated on rc==137"

    def test_wrapper_retry_is_operator_disableable(self) -> None:
        # PHASE1_CI_OOM_RETRY=0 must skip the retry (no auto-retry loop the
        # operator can't turn off).
        src = _jfv()
        assert '"${PHASE1_CI_OOM_RETRY}" == "1"' in src, (
            "retry must be gated behind the env kill-switch"
        )

    def test_wrapper_retry_sets_slim_mode(self) -> None:
        # The retry attempt MUST inject MOUSEDROID_CI_SLIM=1 so ci.sh skips
        # the memory-heaviest stages.
        src = _jfv()
        # Find the retry docker exec — must carry the SLIM env var.
        retry_block_start = src.find("OOM detected")
        assert retry_block_start != -1
        retry_block = src[retry_block_start : retry_block_start + 800]
        assert "MOUSEDROID_CI_SLIM=1" in retry_block, "OOM retry must set MOUSEDROID_CI_SLIM=1"

    def test_wrapper_records_warn_on_retry_success(self) -> None:
        # Successful retry MUST record WARN (not silent PASS) so the operator
        # sees the OOM-driven degradation in the summary.
        src = _jfv()
        assert 'record WARN "static CI (ci.sh, container)"' in src, (
            "retry success must WARN, not silently PASS"
        )


class TestCiShSlimModeContract:
    """ci.sh's slim mode must skip Perf + Regression + E2E when SLIM=1."""

    def test_slim_gates_performance_stage(self) -> None:
        # The Performance stage must be inside a `if MOUSEDROID_CI_SLIM ... else`.
        src = _ci()
        perf_idx = src.find('echo "=== Performance Tests')
        assert perf_idx != -1, "Performance stage marker must exist"
        # Look backwards ~200 chars for the slim conditional.
        context_before = src[max(0, perf_idx - 300) : perf_idx]
        assert "MOUSEDROID_CI_SLIM" in context_before, (
            "Performance stage must be inside a MOUSEDROID_CI_SLIM conditional"
        )

    def test_slim_gates_regression_stage(self) -> None:
        src = _ci()
        reg_idx = src.find('echo "=== Regression Tests')
        assert reg_idx != -1
        context_before = src[max(0, reg_idx - 400) : reg_idx]
        assert "MOUSEDROID_CI_SLIM" in context_before

    def test_slim_gates_e2e_stage(self) -> None:
        src = _ci()
        e2e_idx = src.find('echo "=== E2E Tests')
        assert e2e_idx != -1
        context_before = src[max(0, e2e_idx - 300) : e2e_idx]
        assert "MOUSEDROID_CI_SLIM" in context_before

    def test_slim_off_is_default_backwards_compatible(self) -> None:
        # `${MOUSEDROID_CI_SLIM:-0}` — default MUST be "0" (off) so an operator
        # who has never heard of slim mode gets pre-feature behavior.
        assert "${MOUSEDROID_CI_SLIM:-0}" in _ci(), (
            "MOUSEDROID_CI_SLIM must default to 0 (backwards-compatible)"
        )

    def test_slim_does_not_skip_unit_property_integration(self) -> None:
        # The core signal (Unit+Property+Integration+coverage) MUST run in both
        # modes. If it disappears, coverage semantics break.
        src = _ci()
        unit_idx = src.find('echo "=== Unit + Property + Integration')
        assert unit_idx != -1
        # Look forward ~200 chars for the pytest invocation (not gated by SLIM).
        block = src[unit_idx : unit_idx + 500]
        assert "pytest tests/unit tests/property tests/integration" in block
        # And the enclosing context must NOT be an `if MOUSEDROID_CI_SLIM` block.
        context_before = src[max(0, unit_idx - 200) : unit_idx]
        assert 'if [[ "${MOUSEDROID_CI_SLIM' not in context_before, (
            "Unit+Property+Integration must run in both modes"
        )


class TestCiShSyntax:
    """A stray edit that breaks bash syntax would crash Phase-1 on the rover."""

    def test_ci_sh_parses(self) -> None:
        import subprocess

        subprocess.run(["bash", "-n", _CI.relative_to(_REPO_ROOT).as_posix()], check=True)

    def test_jetson_full_validation_sh_parses(self) -> None:
        import subprocess

        subprocess.run(["bash", "-n", _JFV.relative_to(_REPO_ROOT).as_posix()], check=True)
