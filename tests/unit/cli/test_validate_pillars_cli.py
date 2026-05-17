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
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
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
