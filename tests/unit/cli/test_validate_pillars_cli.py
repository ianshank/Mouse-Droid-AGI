"""Smoke-pass: validate_pillars CLI argparse + exit-code tests."""

from __future__ import annotations

import json
import re

import pytest

from mousedroid.cli.validate_pillars import main


def _extract_json_block(captured_out: str) -> dict[str, object]:
    """Pull the JSON document out of mixed stdout (structlog + JSON).

    The CLI calls ``sys.stdout.write(report.model_dump_json(indent=2))``
    AFTER structured-log events have already gone to stdout. We extract
    the trailing JSON object via a regex anchored on ``{ ... }`` at the
    end of the stream — the pretty-printed Pydantic dump is the only
    multi-line balanced block in the output.
    """
    # Find the LAST top-level JSON object in the stream.
    match = re.search(r"(\{[\s\S]*\})\s*$", captured_out.strip())
    if match is None:
        msg = f"no JSON object found in captured stdout: {captured_out!r}"
        raise AssertionError(msg)
    return json.loads(match.group(1))


def test_dry_run_returns_exit_code_0(capsys: pytest.CaptureFixture[str]) -> None:
    """`--dry-run` exits 0 because every pillar is SKIPPED (not FAIL)."""
    rc = main(["--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "overall=ok" in captured.out


def test_dry_run_with_json_emits_parseable_json(capsys: pytest.CaptureFixture[str]) -> None:
    """`--json --dry-run` writes a valid JSON document to stdout."""
    rc = main(["--dry-run", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = _extract_json_block(captured.out)
    assert "results" in payload
    assert len(payload["results"]) == 10  # all 10 pillars


def test_pillars_filter_restricts_dispatch(capsys: pytest.CaptureFixture[str]) -> None:
    """`--pillars safety,world_model` runs only the two named pillars."""
    rc = main(["--dry-run", "--pillars", "safety,world_model", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = _extract_json_block(captured.out)
    names = {r["name"] for r in payload["results"]}
    assert names == {"safety", "world_model"}


def test_invalid_pillar_filter_returns_empty_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown pillar name yields an empty results list + overall OK (vacuous)."""
    rc = main(["--dry-run", "--pillars", "does_not_exist", "--json"])
    assert rc == 0  # empty result set has no FAIL entries
    captured = capsys.readouterr()
    payload = _extract_json_block(captured.out)
    assert payload["results"] == []


def test_main_without_argv_uses_sys_argv_default(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling ``main()`` without argv parses ``sys.argv`` (operator runbook flow)."""
    monkeypatch.setattr("sys.argv", ["validate_pillars", "--dry-run"])
    rc = main()
    assert rc == 0


def test_help_flag_exits_zero_and_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--help` exits 0 + prints argparse usage (no crash)."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "validate_pillars" in captured.out or "usage" in captured.out.lower()


def test_cli_exits_0_when_overall_status_is_degraded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """DEGRADED (WARN-only, no FAIL) must exit 0 — mirrors preflight CLI contract.

    Reviewer caught a bug where the CLI exited 1 on DEGRADED. CI uses
    these exit codes as the canonical "is the pillar dispatch broken?"
    signal, so DEGRADED → 1 would spuriously fail otherwise-clean runs.
    Pinned by patching ``validate_all_pillars`` to return a synthetic
    DEGRADED report and asserting ``main(...) == 0``.
    """
    from mousedroid.cli import validate_pillars as _cli
    from mousedroid.validation.pillars import (
        PillarReport,
        PillarResult,
        PillarStatus,
    )

    degraded_report = PillarReport(
        results=[
            PillarResult(
                name="curiosity",
                status=PillarStatus.WARN,
                detail="optional subsystem warning",
                elapsed_s=0.0,
            ),
        ],
    )

    async def _fake_validate(*_args: object, **_kwargs: object) -> PillarReport:
        return degraded_report

    monkeypatch.setattr(_cli, "validate_all_pillars", _fake_validate)
    # Use text render (not --json) so the "overall=degraded" line that
    # ``PillarReport.render_text()`` produces shows up in stdout — the
    # JSON dump excludes the ``overall_status`` computed property by
    # default.
    rc = _cli.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "degraded" in captured.out.lower()


# ---------------------------------------------------------------------------
# --strict-skips — opt-in gate on skip_reason="environment"
# ---------------------------------------------------------------------------


def _patch_report_with_skip(
    monkeypatch: pytest.MonkeyPatch,
    skip_reason: str,
) -> None:
    """Patch the CLI's validate_all_pillars to return one SKIPPED pillar."""
    from mousedroid.cli import validate_pillars as _cli
    from mousedroid.validation.pillars import (
        PillarReport,
        PillarResult,
        PillarStatus,
    )

    report = PillarReport(
        results=[
            PillarResult(
                name="growth",
                status=PillarStatus.SKIPPED,
                detail=f"synthetic {skip_reason} skip",
                elapsed_s=0.0,
                skip_reason=skip_reason,  # type: ignore[arg-type]
            ),
        ],
    )

    async def _fake_validate(*_args: object, **_kwargs: object) -> PillarReport:
        return report

    monkeypatch.setattr(_cli, "validate_all_pillars", _fake_validate)


def test_strict_skips_fails_on_environment_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """An `environment` skip (pytest absent) exits 1 under --strict-skips.

    This is the lean-runtime guard: Pattern-B pillars silently SKIP when the
    runtime has no pytest, which would otherwise render as a clean pass.
    """
    from mousedroid.cli import validate_pillars as _cli

    _patch_report_with_skip(monkeypatch, "environment")
    assert _cli.main(["--strict-skips"]) == 1


def test_strict_skips_passes_on_config_disabled_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `config_disabled` skip is legitimate and still exits 0 under strict mode.

    ``memory`` and ``curiosity`` SKIP this way on the production overlay
    (no ``memory:`` block → ``MemoryConfig.enabled`` defaults False), so a
    blanket "fail on any SKIP" would false-fail every campaign run.
    """
    from mousedroid.cli import validate_pillars as _cli

    _patch_report_with_skip(monkeypatch, "config_disabled")
    assert _cli.main(["--strict-skips"]) == 0


def test_environment_skip_without_flag_keeps_legacy_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without --strict-skips the documented contract is unchanged (0 on skip)."""
    from mousedroid.cli import validate_pillars as _cli

    _patch_report_with_skip(monkeypatch, "environment")
    assert _cli.main([]) == 0


def test_strict_skips_rejects_dry_run_combination() -> None:
    """--strict-skips with --dry-run is a usage error (argparse exit code 2).

    Dry-run marks every pillar SKIPPED by design, so strict mode over it is
    contradictory — refuse loudly rather than pass vacuously.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["--strict-skips", "--dry-run"])
    assert excinfo.value.code == 2
