"""F-059 credit ledger: ``tools/credit_ledger.py`` unit tests.

The ledger is the guard between an agent that can spend real money and the cap
the operator set: ``charge`` must refuse BEFORE the API call, ``record`` must
append one JSON object per line without ever rewriting history, and the caps and
the ledger path must come from ``.claude/workforce.yaml`` (never from code).
Everything runs against a temporary repo root holding a copy of the config - no
network, no API key, no write outside ``tmp_path``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools import credit_ledger
from tools.claude_hooks.config import DEFAULT_CONFIG_RELPATH, ConfigError, load_config
from tools.credit_ledger import CapExceeded, LedgerError

_REPO_ROOT = Path(__file__).resolve().parents[3]

_LEDGER_RELPATH = "reports/credit/ledger.jsonl"
_CONFIG_BODY = """\
credit:
    caps_usd:
        A: 10
        B: 60
        C: 15
    ledger: reports/credit/ledger.jsonl
"""


def _write_config(root: Path, body: str = _CONFIG_BODY) -> None:
    path = root / DEFAULT_CONFIG_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo root whose workforce.yaml carries the planned caps (A 10 / B 60 / C 15)."""
    _write_config(tmp_path)
    return tmp_path


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ws": "B",
        "run_id": "wsb-0003",
        "target": "PR-B diff @1a2b3c4",
        "model_requested": "opus",
        "model_resolved": "claude-opus-5-5",
        "resolved_via": "models.retrieve",
        "input_tokens": 184233,
        "output_tokens": 9120,
        "cache_read_tokens": 0,
        "cost_usd": 1.33,
        "cost_source": "ResultMessage.total_cost_usd",
        "artifact": "reports/agent_reviewer/2026-10-18/wsb-0003.md",
        "status": "ok",
    }
    base.update(overrides)
    return base


def _seed(root: Path, *rows: dict[str, Any]) -> Path:
    """Write ledger rows directly, bypassing ``record`` (to set up prior spend)."""
    path = root / _LEDGER_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _lines(root: Path) -> list[str]:
    path = root / _LEDGER_RELPATH
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


# --------------------------------------------------------------------------- #
# charge - refuse BEFORE the call
# --------------------------------------------------------------------------- #
def test_ledger_refuses_over_cap(repo: Path) -> None:
    _seed(repo, {"ws": "A", "cost_usd": 9.0})
    with pytest.raises(CapExceeded) as excinfo:
        credit_ledger.charge("A", 1.01, repo_root=repo)
    message = str(excinfo.value)
    assert "'A'" in message
    assert "10" in message  # the cap
    assert "9" in message  # the spend so far


def test_charge_allows_spending_exactly_up_to_the_cap(repo: Path) -> None:
    _seed(repo, {"ws": "A", "cost_usd": 9.0})
    headroom = credit_ledger.charge("A", 1.0, repo_root=repo)
    assert headroom.remaining_usd == pytest.approx(0.0)
    with pytest.raises(CapExceeded):
        credit_ledger.charge("A", 1.01, repo_root=repo)


def test_charge_never_writes_the_ledger(repo: Path) -> None:
    seeded = _seed(repo, {"ws": "A", "cost_usd": 1.0})
    before = seeded.read_bytes()
    credit_ledger.charge("A", 2.0, repo_root=repo)
    assert seeded.read_bytes() == before


def test_charge_counts_only_the_requested_workstream(repo: Path) -> None:
    _seed(repo, {"ws": "B", "cost_usd": 59.0}, {"ws": "A", "cost_usd": 1.0})
    assert credit_ledger.charge("A", 9.0, repo_root=repo).spent_usd == pytest.approx(1.0)
    assert credit_ledger.charge("B", 1.0, repo_root=repo).spent_usd == pytest.approx(59.0)


def test_charge_starts_from_zero_when_the_ledger_does_not_exist(repo: Path) -> None:
    assert not (repo / _LEDGER_RELPATH).exists()
    headroom = credit_ledger.charge("C", 15.0, repo_root=repo)
    assert headroom.spent_usd == 0.0
    assert headroom.cap_usd == 15.0


def test_charge_uses_exact_decimal_arithmetic(repo: Path) -> None:
    """0.1 + 0.2 is 0.30000000000000004 in floats and would trip a 0.3 cap."""
    _write_config(repo, "credit:\n    caps_usd:\n        A: 0.3\n")
    _seed(repo, {"ws": "A", "cost_usd": 0.1}, {"ws": "A", "cost_usd": 0.2})
    credit_ledger.charge("A", 0.0, repo_root=repo)  # exactly at the cap: allowed
    with pytest.raises(CapExceeded):
        credit_ledger.charge("A", 0.01, repo_root=repo)


def test_charge_refuses_a_workstream_with_no_configured_cap(repo: Path) -> None:
    """Fail closed: an unknown workstream must not be able to spend."""
    with pytest.raises(CapExceeded, match="no spend cap"):
        credit_ledger.charge("Z", 0.01, repo_root=repo)


def test_charge_fails_closed_when_the_config_has_no_credit_block(tmp_path: Path) -> None:
    _write_config(tmp_path, "freeze:\n    enabled: true\n")
    with pytest.raises(CapExceeded, match="no spend cap"):
        credit_ledger.charge("A", 0.01, repo_root=tmp_path)


@pytest.mark.parametrize("estimate", [-0.01, math.nan, math.inf, -math.inf])
def test_charge_rejects_estimates_that_could_slip_past_the_comparison(
    repo: Path, estimate: float
) -> None:
    """NaN compares False against everything, so it would pass a naive ``>`` check."""
    with pytest.raises(ValueError, match="estimate_usd"):
        credit_ledger.charge("A", estimate, repo_root=repo)


def test_charge_rejects_a_boolean_estimate(repo: Path) -> None:
    with pytest.raises(ValueError, match="estimate_usd"):
        credit_ledger.charge("A", True, repo_root=repo)


def test_charge_refuses_when_the_workstream_is_already_over_its_cap(repo: Path) -> None:
    _seed(repo, {"ws": "A", "cost_usd": 12.0})
    with pytest.raises(CapExceeded):
        credit_ledger.charge("A", 0.0, repo_root=repo)


# --------------------------------------------------------------------------- #
# The ledger file - append-only JSON lines
# --------------------------------------------------------------------------- #
def test_record_appends_one_json_object_per_line(repo: Path) -> None:
    credit_ledger.record(_entry(run_id="wsb-0001"), repo_root=repo)
    first_line = _lines(repo)[0]
    credit_ledger.record(_entry(run_id="wsb-0002", cost_usd=0.5), repo_root=repo)

    lines = _lines(repo)
    assert len(lines) == 2
    assert lines[0] == first_line, "appending must never rewrite an earlier row"
    assert [json.loads(line)["run_id"] for line in lines] == ["wsb-0001", "wsb-0002"]


def test_record_writes_lf_line_endings_and_a_single_trailing_newline(repo: Path) -> None:
    credit_ledger.record(_entry(), repo_root=repo)
    raw = (repo / _LEDGER_RELPATH).read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b"\r" not in raw, "a tracked JSONL artifact must not pick up CRLF on Windows"


def test_record_row_has_the_planned_shape_and_key_order(repo: Path) -> None:
    row = credit_ledger.record(_entry(), repo_root=repo)

    assert list(row) == [
        "ts",
        "ws",
        "run_id",
        "target",
        "model_requested",
        "model_resolved",
        "resolved_via",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cost_usd",
        "cost_source",
        "cap_usd",
        "cumulative_ws_usd",
        "artifact",
        "status",
    ]
    assert json.loads(_lines(repo)[0]) == row
    assert row["cap_usd"] == 60.0
    assert row["cumulative_ws_usd"] == pytest.approx(1.33)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", row["ts"])


def test_record_cumulative_accumulates_per_workstream(repo: Path) -> None:
    credit_ledger.record(_entry(ws="B", cost_usd=2.5), repo_root=repo)
    credit_ledger.record(_entry(ws="A", cost_usd=0.25, run_id="wsa-0001"), repo_root=repo)
    third = credit_ledger.record(_entry(ws="B", cost_usd=1.5, run_id="wsb-0002"), repo_root=repo)
    assert third["cumulative_ws_usd"] == pytest.approx(4.0)
    assert credit_ledger.spent_usd("A", repo_root=repo) == pytest.approx(0.25)
    assert credit_ledger.spent_usd("B", repo_root=repo) == pytest.approx(4.0)


def test_record_keeps_a_caller_supplied_timestamp(repo: Path) -> None:
    row = credit_ledger.record(_entry(ts="2026-10-18T14:02:11Z"), repo_root=repo)
    assert row["ts"] == "2026-10-18T14:02:11Z"


def test_record_defaults_cache_read_tokens_to_zero(repo: Path) -> None:
    entry = _entry()
    del entry["cache_read_tokens"]
    assert credit_ledger.record(entry, repo_root=repo)["cache_read_tokens"] == 0


def test_record_overwrites_caller_supplied_derived_fields(repo: Path) -> None:
    """``cap_usd`` / ``cumulative_ws_usd`` belong to the ledger, not the caller."""
    row = credit_ledger.record(
        _entry(cap_usd=9999.0, cumulative_ws_usd=0.0, cost_usd=2.0), repo_root=repo
    )
    assert row["cap_usd"] == 60.0
    assert row["cumulative_ws_usd"] == pytest.approx(2.0)


def test_record_creates_the_ledger_directory(tmp_path: Path) -> None:
    _write_config(tmp_path)
    assert not (tmp_path / "reports").exists()
    credit_ledger.record(_entry(), repo_root=tmp_path)
    assert (tmp_path / _LEDGER_RELPATH).is_file()


def test_record_keeps_non_ascii_text_readable(repo: Path) -> None:
    credit_ledger.record(_entry(target="PR-B — diff"), repo_root=repo)
    assert "—" in (repo / _LEDGER_RELPATH).read_text(encoding="utf-8")


def test_record_still_records_a_run_that_overshoots_the_cap(repo: Path) -> None:
    """The money is already spent; hiding it would be worse than recording it."""
    _seed(repo, {"ws": "A", "cost_usd": 9.5})
    row = credit_ledger.record(_entry(ws="A", cost_usd=2.0, run_id="wsa-0009"), repo_root=repo)
    assert row["cumulative_ws_usd"] == pytest.approx(11.5)
    with pytest.raises(CapExceeded):
        credit_ledger.charge("A", 0.0, repo_root=repo)


# --------------------------------------------------------------------------- #
# record - what it refuses to write
# --------------------------------------------------------------------------- #
def test_record_rejects_an_unknown_key(repo: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        credit_ledger.record(_entry(cost=1.0), repo_root=repo)
    assert _lines(repo) == []


@pytest.mark.parametrize(
    "missing",
    ["ws", "run_id", "target", "model_requested", "model_resolved", "cost_usd", "status"],
)
def test_record_rejects_a_missing_required_key(repo: Path, missing: str) -> None:
    entry = _entry()
    del entry[missing]
    with pytest.raises(ValueError, match=missing):
        credit_ledger.record(entry, repo_root=repo)
    assert _lines(repo) == []


def test_record_rejects_a_workstream_with_no_cap(repo: Path) -> None:
    """A row for an uncapped workstream would escape every future cap check."""
    with pytest.raises(ValueError, match="no spend cap"):
        credit_ledger.record(_entry(ws="Z"), repo_root=repo)
    assert _lines(repo) == []


@pytest.mark.parametrize("cost", [-1.0, math.nan, math.inf, "1.33", None, True])
def test_record_rejects_an_invalid_cost(repo: Path, cost: Any) -> None:
    with pytest.raises(ValueError, match="cost_usd"):
        credit_ledger.record(_entry(cost_usd=cost), repo_root=repo)
    assert _lines(repo) == []


@pytest.mark.parametrize("tokens", [-1, 1.5, "100", None])
def test_record_rejects_invalid_token_counts(repo: Path, tokens: Any) -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        credit_ledger.record(_entry(input_tokens=tokens), repo_root=repo)
    assert _lines(repo) == []


def test_record_refuses_a_row_that_contains_an_api_key(repo: Path) -> None:
    """R8: the ledger is a tracked file and must never carry a credential."""
    leaked = "sk-ant-" + "api03-" + "A" * 20
    with pytest.raises(ValueError, match="credential"):
        credit_ledger.record(_entry(target=f"reviewed with {leaked}"), repo_root=repo)
    assert _lines(repo) == []


# --------------------------------------------------------------------------- #
# A damaged ledger fails closed
# --------------------------------------------------------------------------- #
def test_a_corrupt_ledger_line_blocks_spending(repo: Path) -> None:
    path = _seed(repo, {"ws": "A", "cost_usd": 1.0})
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("{not json\n")
    with pytest.raises(LedgerError, match=r"ledger\.jsonl:2"):
        credit_ledger.charge("A", 0.01, repo_root=repo)


def test_a_ledger_row_without_a_numeric_cost_blocks_spending(repo: Path) -> None:
    _seed(repo, {"ws": "A", "cost_usd": "1.0"})
    with pytest.raises(LedgerError, match="cost_usd"):
        credit_ledger.charge("A", 0.01, repo_root=repo)


def test_blank_lines_in_the_ledger_are_ignored(repo: Path) -> None:
    path = _seed(repo, {"ws": "A", "cost_usd": 2.0})
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n\n")
    assert credit_ledger.spent_usd("A", repo_root=repo) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Caps and path come from workforce.yaml - never from code
# --------------------------------------------------------------------------- #
def test_caps_come_from_the_workforce_config_not_from_code(tmp_path: Path) -> None:
    _write_config(tmp_path, "credit:\n    caps_usd:\n        A: 1.5\n")
    assert credit_ledger.charge("A", 1.5, repo_root=tmp_path).cap_usd == 1.5
    with pytest.raises(CapExceeded):
        credit_ledger.charge("A", 1.6, repo_root=tmp_path)


def test_the_ledger_path_comes_from_the_workforce_config(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "credit:\n    caps_usd:\n        A: 5\n    ledger: reports/elsewhere/spend.jsonl\n",
    )
    credit_ledger.record(_entry(ws="A"), repo_root=tmp_path)
    assert (tmp_path / "reports" / "elsewhere" / "spend.jsonl").is_file()
    assert not (tmp_path / _LEDGER_RELPATH).exists()


def test_an_explicit_config_object_overrides_the_file(repo: Path) -> None:
    cfg = load_config(repo_root=repo).credit.model_copy(update={"caps_usd": {"A": 0.5}})
    with pytest.raises(CapExceeded):
        credit_ledger.charge("A", 0.6, config=cfg, repo_root=repo)


def test_an_invalid_workforce_config_stops_the_ledger(tmp_path: Path) -> None:
    _write_config(tmp_path, "credit:\n    caps_usd:\n        A: -5\n")
    with pytest.raises(ConfigError):
        credit_ledger.charge("A", 0.01, repo_root=tmp_path)


def test_the_checked_in_workforce_config_declares_the_planned_caps() -> None:
    """The plan's budget table (A 10 / B 60 / C 15) is a governance decision.

    A change belongs in a reviewed edit of ``.claude/workforce.yaml`` AND here,
    not in a silent drift of one of them.
    """
    raw = yaml.safe_load((_REPO_ROOT / DEFAULT_CONFIG_RELPATH).read_text(encoding="utf-8"))
    assert raw["credit"]["caps_usd"] == {"A": 10, "B": 60, "C": 15}
    assert raw["credit"]["ledger"] == _LEDGER_RELPATH
    loaded = load_config(repo_root=_REPO_ROOT).credit
    assert loaded.caps_usd == {"A": 10.0, "B": 60.0, "C": 15.0}
    assert loaded.ledger == _LEDGER_RELPATH


# --------------------------------------------------------------------------- #
# status CLI
# --------------------------------------------------------------------------- #
def test_status_reports_cap_spent_and_remaining(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(repo, {"ws": "A", "cost_usd": 2.5}, {"ws": "B", "cost_usd": 10.0})
    rc = credit_ledger.main(["status", "--repo-root", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert re.search(r"^A\s+10\.00\s+2\.50\s+7\.50$", out, re.MULTILINE), out
    assert re.search(r"^B\s+60\.00\s+10\.00\s+50\.00$", out, re.MULTILINE), out
    assert re.search(r"^C\s+15\.00\s+0\.00\s+15\.00$", out, re.MULTILINE), out


def test_status_exits_nonzero_when_a_workstream_is_over_its_cap(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(repo, {"ws": "A", "cost_usd": 11.0})
    assert credit_ledger.main(["status", "--repo-root", str(repo)]) == 1
    assert "OVER" in capsys.readouterr().out


def test_status_exits_nonzero_on_a_corrupt_ledger(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "reports" / "credit").mkdir(parents=True)
    (repo / _LEDGER_RELPATH).write_text("{not json\n", encoding="utf-8")
    assert credit_ledger.main(["status", "--repo-root", str(repo)]) == 1
    assert "ledger.jsonl:1" in capsys.readouterr().err
