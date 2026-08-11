"""Regression: jetson_full_validation.sh hardening contract (PR #116).

Pins the no-hardcoded-values + parse-clean guarantees so a future edit that
re-introduces a literal timeout/port/namespace, or breaks the shell syntax,
fails loudly. Also asserts the PR #116 deliverable files exist (rename guard).

These are static/text checks plus an optional ``bash -n`` parse — they run on
any host (the ``bash -n`` step skips gracefully where bash is absent).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "jetson_full_validation.sh"
_RUNBOOK = _REPO_ROOT / "docs" / "runbooks" / "jetson-full-validation.md"
_LIVE_TEST = _REPO_ROOT / "tests" / "hardware" / "test_llm_gateway_metrics_live_jetson.py"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_deliverable_files_exist() -> None:
    """Rename guard for the PR #116 deliverables."""
    assert _SCRIPT.is_file(), f"missing wrapper: {_SCRIPT}"
    assert _RUNBOOK.is_file(), f"missing runbook: {_RUNBOOK}"
    assert _LIVE_TEST.is_file(), f"missing live-metrics test: {_LIVE_TEST}"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash execution test")
def test_script_parses_clean() -> None:
    """`bash -n` parse check (skips where bash is unavailable, e.g. some CI)."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH")
    result = subprocess.run(
        [bash, "-n", str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


# Literals that would indicate a hardcoded tunable crept back in. Each tunable
# must instead read an env var with a default (e.g. "${VAR:-N}").
_HARDCODED_PATTERNS = (
    r"--max-time\s+[0-9]",
    r"--timeout=[0-9]",
    r"--duration\s+[0-9]",
    r"--tail\s+[0-9]",
    r"seq\s+1\s+[0-9]",
)


@pytest.mark.parametrize("pattern", _HARDCODED_PATTERNS)
def test_no_hardcoded_tunables(pattern: str) -> None:
    """No literal timeout/retry/duration/tail values — all must be env-driven."""
    matches = re.findall(pattern, _script_text())
    assert not matches, f"hardcoded literal matching {pattern!r}: {matches}"


def test_namespace_is_config_derived_not_literal() -> None:
    """The /metrics grep must use the derived ${NAMESPACE}, not a literal prefix."""
    text = _script_text()
    assert "MOUSEDROID_METRICS__NAMESPACE" in text
    # The hardcoded 'mousedroid_' prefix must not appear in a grep pattern.
    assert "'^mousedroid_'" not in text
    assert 'grep -q "^${NAMESPACE}_"' in text


@pytest.mark.parametrize(
    "env_var",
    [
        # Path / target tunables.
        "MOUSEDROID_SMOKE_CONTAINER",
        "MOUSEDROID_VALIDATION_REPORT_ROOT",
        "MOUSEDROID_TELEMETRY_URL",
        "MOUSEDROID_JETSON_CONFIG",
        "MOUSEDROID_LIDAR_PROBE_PORT",
        "MOUSEDROID_VALIDATION_MISSION",
        "VENV_DIR",
        "MOUSEDROID_METRICS__NAMESPACE",
        # Timing / retry tunables.
        "MOUSEDROID_VALIDATION_HEALTH_RETRIES",
        "MOUSEDROID_VALIDATION_HEALTH_INTERVAL_S",
        "MOUSEDROID_VALIDATION_HTTP_TIMEOUT_S",
        "MOUSEDROID_VALIDATION_PYTEST_TIMEOUT_S",
        "MOUSEDROID_VALIDATION_LIDAR_DURATION_S",
        "MOUSEDROID_VALIDATION_LOG_TAIL",
        # Trend journal (F-018).
        "MOUSEDROID_VALIDATION_JOURNAL",
        "MOUSEDROID_VALIDATION_JOURNAL_MAX_BYTES",
        "MOUSEDROID_VALIDATION_TREND_SLOW_RATIO",
        "MOUSEDROID_VALIDATION_TREND_SLOW_FLOOR_S",
    ],
)
def test_documented_env_tunables_present(env_var: str) -> None:
    """Every tunable is wired (env override with a default) AND documented."""
    text = _script_text()
    assert f"{env_var}:-" in text, f"{env_var} not wired with a default"
    assert text.count(env_var) >= 2, f"{env_var} not documented in the header"


# Patterns that would constitute a secret leak. Broader than the literal
# ``${VAR}`` form so unbraced refs, printf/log/printenv shapes are also
# caught (CodeRabbit PR #117).
_SECRET_LEAK_PATTERN = re.compile(
    r"(?m)^\s*(?:echo|printf|log|printenv)\b[^\n]*\b"
    r"(?:ANTHROPIC_API_KEY|MOUSEDROID_TELEMETRY_TOKEN)\b"
)


def test_secrets_presence_checked_only() -> None:
    """Secrets must never be echoed — only their presence is tested."""
    text = _script_text()
    leak = _SECRET_LEAK_PATTERN.search(text)
    assert leak is None, f"secret-leak shape detected: {leak.group(0)!r}"
    # The strict interpolation form is also rejected (defense in depth — catches
    # cases where the secret appears outside a recognised log/echo verb).
    assert "${ANTHROPIC_API_KEY}" not in text
    assert "${MOUSEDROID_TELEMETRY_TOKEN}" not in text


class TestTrendThreading:
    """F-018: Phase-2 preflight appends to the trend journal + summary shim."""

    def test_phase2_preflight_threads_journal_flags(self) -> None:
        text = _script_text()
        assert '--journal-path "${TREND_JOURNAL}"' in text
        assert "--trend" in text
        assert '--journal-max-bytes "${TREND_JOURNAL_MAX_BYTES}"' in text

    def test_journal_lives_under_report_root_not_run_dir(self) -> None:
        # A per-run journal would never accumulate the >=2 runs a trend needs.
        text = _script_text()
        assert "${REPORT_ROOT}/trend_journal.jsonl" in text
        assert "${RUN_DIR}/trend_journal" not in text

    def test_summary_uses_renderer_with_bash_fallback(self) -> None:
        text = _script_text()
        assert "scripts/render_validation_summary.py" in text
        assert "write_summary_fallback" in text, "python-less hosts must still get a summary"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash execution test")
class TestSummaryFallbackExecution:
    """Gap-analysis: prove the python-less fallback actually produces SUMMARY.md.

    Extracts the real ``log``/``write_summary_fallback``/``write_summary``
    function bodies from the script (column-0 ``name() {`` … ``}`` blocks) and
    runs them in a bash harness where ``resolve_host_python`` fails — the
    exact "python-less host" condition the fallback exists for.
    """

    @staticmethod
    def _extract_function(source: str, name: str) -> str:
        lines = source.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}() {{"))
        end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
        return "\n".join(lines[start : end + 1])

    def test_fallback_writes_summary_when_python_unavailable(self, tmp_path: Path) -> None:
        source = _script_text()
        # log() is a one-liner delegating to ts(); stub it in the harness and
        # extract only the two multi-line summary functions under test.
        functions = "\n\n".join(
            self._extract_function(source, name)
            for name in ("write_summary_fallback", "write_summary")
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        harness = tmp_path / "harness.sh"
        harness.write_text(
            "#!/usr/bin/env bash\n"
            "set -uo pipefail\n"
            'log() { echo "$*"; }\n'
            f"{functions}\n"
            "# python-less host: the renderer path must fail over cleanly.\n"
            "resolve_host_python() { return 1; }\n"
            f'RUN_DIR="{run_dir}"\n'
            'STAMP="20260703T000000Z"\n'
            'REPO_DIR="/opt/mousedroid"\n'
            'PROD_CONFIG="config/jetson_production.yaml"\n'
            'TELEMETRY_URL="http://127.0.0.1:8080"\n'
            "PASSES=1 WARNS=1 FAILURES=1\n"
            'RESULTS=("PASS|preflight (real)|" "WARN|serial smoke|dead ESP32"'
            ' "FAIL|renderer|exit 1|see log")\n'
            "write_summary\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["bash", str(harness)], capture_output=True, text=True, cwd=_REPO_ROOT
        )
        assert proc.returncode == 0, proc.stderr
        summary = (run_dir / "SUMMARY.md").read_text(encoding="utf-8")
        assert "| PASS | preflight (real) |" in summary
        assert "| WARN | serial smoke | dead ESP32 |" in summary
        # A |-bearing note must render as ONE escaped row, not extra columns
        # (mirrors _escape_cell in mousedroid/validation/summary.py).
        assert "| FAIL | renderer | exit 1\\|see log |" in summary
        assert "Totals: PASS=1 WARN=1 FAIL=1" in summary
        assert "fallback" in proc.stdout, "the fallback path must announce itself"
