"""F-059 WS-C: ``tools/dead_code_triage_batch.py`` unit tests.

The tool sends source excerpts to a paid API, so the contract worth pinning is
what it does BEFORE it spends (price the batch from the workforce config, refuse
over a cap, refuse files that must never leave the machine) and what it does
AFTER (match results to requests by ``custom_id``, turn anything it cannot trust
into ``verify``, record the real usage once).

Everything runs against a throw-away repo under ``tmp_path`` and a fake client:
no network, no API key, no write outside ``tmp_path``. The last group drives the
REAL ``anthropic`` SDK over a mock transport and skips when the optional extra is
not installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import types
import typing
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from tools import credit_ledger
from tools import dead_code_triage_batch as triage
from tools.claude_hooks.config import (
    DEFAULT_CONFIG_RELPATH,
    BatchTriageConfig,
    ModelPrice,
    WorkforceConfig,
    load_config,
)
from tools.claude_hooks.secret_scan import ScanOutcome
from tools.credit_ledger import CapExceeded

_SRC = "src/mousedroid"
_CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # the Message Batches API's own rule
_FIXED_DATE = "2026-10-19"
_WORKFORCE_YAML = """\
credit:
    caps_usd:
        A: 10
        B: 60
        C: 15
    ledger: reports/credit/ledger.jsonl
    prices:
        batch_multiplier: 0.5
        models:
            claude-sonnet-5-5:
                input_usd_per_mtok: 2.0
                output_usd_per_mtok: 10.0
"""


# --------------------------------------------------------------------------- #
# A throw-away repo, audit rows, a fake client
# --------------------------------------------------------------------------- #
def _numbered_source(lines: int) -> str:
    """A file whose line N says ``line N`` - so a test can see exactly which lines were sent."""
    return "".join(f"value_{n} = {n}  # line {n}\n" for n in range(1, lines + 1))


_SERVICE_HEAD = '''\
"""Alpha service."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, field_validator, model_validator


@runtime_checkable
class WidgetProtocol(Protocol):
    """Structural interface: callers hold the Protocol, never the concrete class."""

    label: str

    def render(self) -> str: ...

    async def refresh(self) -> None: ...


class Settings(BaseModel):
    size: int = 3

    @field_validator("size")
    @classmethod
    def _check_size(cls, value: int) -> int:
        return value

    @model_validator(mode="after")
    def _check_all(self) -> "Settings":
        return self


def orphan_helper(x: int) -> int:
    return x
'''


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _lineno(source: str, needle: str) -> int:
    for number, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return number
    raise LookupError(f"{needle!r} is not in the fixture source")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo root: workforce config, three packages' worth of source, one env file."""
    _write(tmp_path, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML)
    _write(tmp_path, f"{_SRC}/alpha/__init__.py", "")
    _write(tmp_path, f"{_SRC}/alpha/service.py", _SERVICE_HEAD)
    _write(tmp_path, f"{_SRC}/alpha/util.py", _numbered_source(90))
    _write(tmp_path, f"{_SRC}/beta/engine.py", _numbered_source(40))
    _write(tmp_path, f"{_SRC}/gamma.py", _numbered_source(12))
    _write(tmp_path, "config/docker.env", "EXAMPLE_SETTING=not-secret\n")
    return tmp_path


def _row(
    filename: str, lineno: int, name: str, kind: str = "function", confidence: int = 60
) -> dict[str, Any]:
    return {
        "filename": filename,
        "lineno": lineno,
        "name": name,
        "type": kind,
        "confidence": confidence,
    }


def _finding(
    filename: str, lineno: int, name: str, kind: str = "function", confidence: int = 60
) -> triage.Finding:
    return triage.Finding.from_row(_row(filename, lineno, name, kind, confidence))


def _audit_rows() -> list[dict[str, Any]]:
    """Rows in the three shapes real audits come in: Windows-absolute, POSIX-absolute, relative."""
    win = "C:\\Users\\dev\\proj\\src\\mousedroid"
    orphan = _lineno(_SERVICE_HEAD, "def orphan_helper")
    render = _lineno(_SERVICE_HEAD, "def render")
    return [
        _row(f"{win}\\alpha\\service.py", orphan, "orphan_helper"),
        _row(f"{win}\\alpha\\service.py", render, "render", "method"),
        _row("/home/ci/work/src/mousedroid/alpha/util.py", 30, "value_30", "variable"),
        _row("/home/ci/work/src/mousedroid/alpha/util.py", 70, "value_70", "variable", 100),
        _row(f"{_SRC}/beta/engine.py", 20, "value_20", "variable"),
        _row("/home/ci/work/src/mousedroid/gamma.py", 5, "value_5", "variable"),
    ]


def _write_findings(repo: Path, rows: list[dict[str, Any]] | None = None) -> Path:
    path = repo / "reports" / "dead_code" / f"{_FIXED_DATE}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _audit_rows() if rows is None else rows
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _workforce(repo: Path) -> WorkforceConfig:
    return load_config(repo_root=repo)


def _settings(repo: Path) -> BatchTriageConfig:
    return _workforce(repo).credit.batch_triage


def _clean_scanner(_text: str) -> ScanOutcome:
    return ScanOutcome(clean=True, available=True)


def _flagging_scanner(_text: str) -> ScanOutcome:
    return ScanOutcome(clean=False, available=True, detail="1 leak")


def _missing_scanner(_text: str) -> ScanOutcome:
    return ScanOutcome(clean=True, available=False, detail="'gitleaks' not found on PATH")


class _Batches:
    """``client.messages.batches``: records ``create``, replays statuses and results."""

    def __init__(
        self,
        statuses: Iterable[str] = ("ended",),
        results: Iterable[Any] = (),
        create_error: Exception | None = None,
    ) -> None:
        self.created: list[list[dict[str, Any]]] = []
        self.statuses = list(statuses)
        self.results_to_serve = list(results)
        self.create_error = create_error
        self.retrieve_calls = 0

    def create(self, *, requests: Iterable[dict[str, Any]]) -> Any:
        if self.create_error is not None:
            raise self.create_error
        self.created.append(list(requests))
        return types.SimpleNamespace(id="msgbatch_test01", processing_status="in_progress")

    def retrieve(self, batch_id: str) -> Any:
        self.retrieve_calls += 1
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return types.SimpleNamespace(id=batch_id, processing_status=status)

    def results(self, batch_id: str) -> Iterable[Any]:
        return iter(self.results_to_serve)


class _FakeClient:
    def __init__(
        self,
        *,
        resolved: str = "claude-sonnet-5-5",
        retrieve_error: Exception | None = None,
        statuses: Iterable[str] = ("ended",),
        results: Iterable[Any] = (),
        create_error: Exception | None = None,
    ) -> None:
        self.models_asked: list[str] = []
        self.options_used: list[dict[str, Any]] = []

        def _retrieve(model_id: str) -> Any:
            self.models_asked.append(model_id)
            if retrieve_error is not None:
                raise retrieve_error
            return types.SimpleNamespace(id=resolved, lifecycle="active")

        self.models = types.SimpleNamespace(retrieve=_retrieve)
        self.batches = _Batches(statuses, results, create_error)
        self.messages = types.SimpleNamespace(batches=self.batches)

    def with_options(self, **options: Any) -> _FakeClient:
        """Like the SDK's: a client with per-call options (here, the same one, recorded)."""
        self.options_used.append(options)
        return self


def _usage(input_tokens: int, output_tokens: int, cache_read: int = 0) -> Any:
    return types.SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=0,
    )


def _succeeded(
    custom_id: str,
    text: str,
    *,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    stop_reason: str = "end_turn",
) -> Any:
    message = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=_usage(input_tokens, output_tokens),
    )
    return types.SimpleNamespace(
        custom_id=custom_id, result=types.SimpleNamespace(type="succeeded", message=message)
    )


def _errored(custom_id: str, kind: str = "overloaded_error", message: str = "Overloaded") -> Any:
    error = types.SimpleNamespace(
        type="error",
        error=types.SimpleNamespace(type=kind, message=message),
        request_id="req_1",
    )
    return types.SimpleNamespace(
        custom_id=custom_id, result=types.SimpleNamespace(type="errored", error=error)
    )


def _ended_without_answer(custom_id: str, kind: str) -> Any:
    return types.SimpleNamespace(custom_id=custom_id, result=types.SimpleNamespace(type=kind))


def _entry(finding: triage.Finding, verdict: str = "keep", reason: str = "kept") -> dict[str, Any]:
    return {
        "name": finding.name,
        "filename": finding.filename,
        "lineno": finding.lineno,
        "verdict": verdict,
        "reason": reason,
    }


def _answer(entries: list[dict[str, Any]]) -> str:
    return json.dumps({"findings": entries})


def _answer_all(
    planned: triage.PlannedRequest,
    verdicts: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """A succeeded result answering every finding of ``planned`` (default verdict ``keep``)."""
    chosen = verdicts or {}
    entries = [
        _entry(f, chosen.get(f.name, "keep"), f"reason for {f.name}") for f in planned.findings
    ]
    return _succeeded(planned.custom_id, _answer(entries), **kwargs)


class _Clock:
    """Injected ``sleep`` / ``monotonic``: waiting advances the clock, nothing really sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the 'UTC now' the tool stamps into records and documents."""
    monkeypatch.setattr(
        triage, "_utc_now", lambda: datetime(2026, 10, 19, 14, 2, 11, tzinfo=timezone.utc)
    )


_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def _connect_target(address: Any) -> str:
    return str(address[0]) if isinstance(address, tuple) else str(address)


@pytest.fixture(autouse=True)
def _no_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a socket to anywhere but loopback.

    Every test here is meant to run with no network at all: fakes, or a mock HTTP
    transport on ``mock.invalid``. This turns "meant to" into a failure the day one
    of them reaches for a real host (an SDK default, a forgotten ``base_url``).
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _guard(address: Any) -> None:
        if _connect_target(address) not in _LOOPBACK:
            raise RuntimeError(f"a test tried to connect to {address!r}")

    def _connect(self: socket.socket, address: Any) -> None:
        _guard(address)
        real_connect(self, address)

    def _connect_ex(self: socket.socket, address: Any) -> int:
        _guard(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _connect_ex)


def _submit(repo: Path, client: Any | None = None, **kwargs: Any) -> triage.SubmitOutcome:
    kwargs.setdefault("scanner", _clean_scanner)
    findings = kwargs.pop("findings_path", None) or _write_findings(repo)
    return triage.submit(
        findings,
        kwargs.pop("cap_usd", 15.0),
        repo_root=repo,
        client=_FakeClient() if client is None else client,
        **kwargs,
    )


def _plan(repo: Path, rows: list[dict[str, Any]] | None = None) -> triage.Plan:
    findings, _sha = triage.load_findings(_write_findings(repo, rows))
    return triage.make_plan(findings, _settings(repo))


def _record(repo: Path) -> dict[str, Any]:
    path = repo / "reports" / "credit" / "batches" / f"{_FIXED_DATE}.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


# --------------------------------------------------------------------------- #
# Findings: loading and chunking
# --------------------------------------------------------------------------- #
def test_load_findings_normalises_paths_and_hashes_the_raw_bytes(repo: Path) -> None:
    path = _write_findings(repo)
    findings, sha = triage.load_findings(path)

    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert {f.filename for f in findings} == {
        f"{_SRC}/alpha/service.py",
        f"{_SRC}/alpha/util.py",
        f"{_SRC}/beta/engine.py",
        f"{_SRC}/gamma.py",
    }
    assert all("\\" not in f.filename and ":" not in f.filename for f in findings)


@pytest.mark.parametrize(
    "payload",
    [
        {"not": "a list"},
        ["not a row"],
        [{"filename": "src/mousedroid/a.py", "lineno": 1, "name": "x", "type": "function"}],
        [_row("src/mousedroid/a.py", True, "x")],
        [_row("src/mousedroid/a.py", "3", "x")],
        [_row("src/mousedroid/a.py", 0, "x")],
        [_row("src/mousedroid/a.py", 1, "")],
        [_row("src/mousedroid/a.py", 1, "x", confidence=101)],
        [_row("", 1, "x")],
    ],
)
def test_load_findings_rejects_a_malformed_audit_instead_of_guessing(
    tmp_path: Path, payload: Any
) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(triage.FindingsError):
        triage.load_findings(path)


def test_load_findings_rejects_a_file_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(triage.FindingsError, match="not valid JSON"):
        triage.load_findings(path)


def test_chunking_groups_by_top_level_package() -> None:
    findings = [
        _finding("C:\\Users\\dev\\proj\\src\\mousedroid\\alpha\\service.py", 10, "a"),
        _finding("/home/ci/work/src/mousedroid/alpha/util.py", 3, "b"),
        _finding("src/mousedroid/beta/engine.py", 7, "c"),
        _finding("/home/ci/work/src/mousedroid/gamma.py", 2, "d"),
        _finding("C:\\Users\\dev\\proj\\src\\mousedroid\\beta\\deep\\nested\\mod.py", 1, "e"),
    ]

    chunks = triage.chunk_by_module(findings)

    assert list(chunks) == [f"{_SRC}/alpha", f"{_SRC}/beta", f"{_SRC}/gamma.py"]
    assert [f.name for f in chunks[f"{_SRC}/alpha"]] == ["a", "b"]
    assert [f.name for f in chunks[f"{_SRC}/beta"]] == ["e", "c"]  # deep/nested sorts before engine
    assert [f.name for f in chunks[f"{_SRC}/gamma.py"]] == ["d"]


def test_chunking_does_not_depend_on_input_order() -> None:
    findings = [_finding(f"src/mousedroid/p{n % 3}/m.py", n, f"s{n}") for n in range(1, 13)]
    assert triage.chunk_by_module(findings) == triage.chunk_by_module(list(reversed(findings)))


def test_a_path_that_repeats_the_source_root_is_cut_at_the_last_one() -> None:
    nested = _finding("/work/src/mousedroid/clone/src/mousedroid/arm/x.py", 1, "x")
    assert nested.filename == f"{_SRC}/arm/x.py"


# --------------------------------------------------------------------------- #
# The plan: requests, ids, splitting, withheld findings
# --------------------------------------------------------------------------- #
def test_plan_has_one_request_per_package_with_api_valid_unique_ids(repo: Path) -> None:
    plan = _plan(repo)

    ids = [r.custom_id for r in plan.requests]
    assert ids == ["dead-alpha", "dead-beta", "dead-gamma_py"]
    assert all(_CUSTOM_ID.match(i) for i in ids)
    assert len(set(ids)) == len(ids)
    assert sum(len(r.findings) for r in plan.requests) == len(_audit_rows())


def test_plan_splits_an_oversize_package_and_keeps_every_id_valid(repo: Path) -> None:
    rows = [_row(f"{_SRC}/alpha/util.py", n, f"value_{n}", "variable") for n in range(1, 46)]
    plan = _plan(repo, rows)

    assert [r.custom_id for r in plan.requests] == ["dead-alpha", "dead-alpha-2", "dead-alpha-3"]
    assert [len(r.findings) for r in plan.requests] == [20, 20, 5]
    assert all(_CUSTOM_ID.match(r.custom_id) for r in plan.requests)
    # nothing lost, nothing sent twice, order kept
    sent = [f.lineno for r in plan.requests for f in r.findings]
    assert sent == list(range(1, 46))


def test_plan_follows_the_configured_request_size(repo: Path) -> None:
    _write(
        repo,
        DEFAULT_CONFIG_RELPATH,
        _WORKFORCE_YAML + "    batch_triage:\n        max_findings_per_request: 2\n",
    )
    rows = [_row(f"{_SRC}/alpha/util.py", n, f"value_{n}", "variable") for n in range(1, 6)]
    assert [len(r.findings) for r in _plan(repo, rows).requests] == [2, 2, 1]


def test_custom_id_is_prefixed_and_sanitised() -> None:
    assert triage.custom_id_for("config") == "dead-config"
    assert triage.custom_id_for(f"{_SRC}/config") == "dead-config"
    assert triage.custom_id_for(f"{_SRC}/factory.py") == "dead-factory_py"
    assert triage.custom_id_for("a b/c") == "dead-a_b_c"


def test_custom_id_longer_than_the_api_allows_is_an_error_not_a_truncation() -> None:
    with pytest.raises(triage.TriageError, match="64"):
        triage.custom_id_for("x" * 80)


def test_plan_refuses_two_packages_that_sanitise_to_the_same_id(repo: Path) -> None:
    rows = [_row(f"{_SRC}/a.b.py", 1, "x"), _row(f"{_SRC}/a_b.py", 1, "y")]
    with pytest.raises(triage.TriageError, match="same custom_id"):
        _plan(repo, rows)


@pytest.mark.parametrize(
    "filename",
    [
        "config/docker.env",
        "src/mousedroid/settings.env",
        "docker.env.production",
        ".env.local",
        "../outside/secrets.py",
        "/etc/passwd",
        "src/mousedroid/alpha/notes.txt",
    ],
)
def test_plan_withholds_findings_the_prompt_builder_would_refuse(repo: Path, filename: str) -> None:
    safe = _row(f"{_SRC}/beta/engine.py", 20, "value_20", "variable")
    plan = _plan(repo, [safe, _row(filename, 3, "leaky")])

    assert [f.name for r in plan.requests for f in r.findings] == ["value_20"]
    assert [w.finding.name for w in plan.withheld] == ["leaky"]
    assert plan.withheld[0].reason  # said why, never silently dropped


def test_plan_fingerprint_is_stable_and_follows_what_would_change_the_mapping(repo: Path) -> None:
    first = _plan(repo)
    again = _plan(repo)
    assert first.fingerprint() == again.fingerprint()

    _write(
        repo,
        DEFAULT_CONFIG_RELPATH,
        _WORKFORCE_YAML + "    batch_triage:\n        max_findings_per_request: 1\n",
    )
    assert _plan(repo).fingerprint() != first.fingerprint()


# --------------------------------------------------------------------------- #
# The request: excerpt, hints, params
# --------------------------------------------------------------------------- #
def _user_text(request: dict[str, Any]) -> str:
    content = request["params"]["messages"][0]["content"]
    return content if isinstance(content, str) else "".join(block["text"] for block in content)


def test_request_carries_excerpt_and_protocol_hints(repo: Path) -> None:
    orphan = _lineno(_SERVICE_HEAD, "def orphan_helper")
    render = _lineno(_SERVICE_HEAD, "def render")
    findings = [
        _finding(f"{_SRC}/alpha/service.py", orphan, "orphan_helper"),
        _finding(f"{_SRC}/alpha/service.py", render, "render", "method"),
    ]

    request = triage.build_request(f"{_SRC}/alpha", findings, repo)
    text = _user_text(request)

    # the excerpt: the finding's own source line, with line numbers
    assert "def orphan_helper(x: int) -> int:" in text
    assert re.search(rf"{orphan}\s*\|\s*def orphan_helper", text)
    # the package's Protocol members and pydantic validators, by name
    assert "- WidgetProtocol: label, refresh, render" in text
    assert "Settings._check_size (field_validator)" in text
    assert "Settings._check_all (model_validator)" in text
    # and the finding list itself
    assert "orphan_helper" in text
    assert "vulture confidence 60" in text
    assert request["custom_id"] == "dead-alpha"


def test_request_excerpt_is_the_configured_window_clamped_to_the_file(repo: Path) -> None:
    mid = _finding(f"{_SRC}/alpha/util.py", 45, "value_45", "variable")
    text = _user_text(triage.build_request(f"{_SRC}/alpha", [mid], repo))
    assert "value_25 = 25" in text  # 45 +- 20 inclusive
    assert "value_65 = 65" in text
    assert "value_24 = 24" not in text
    assert "value_66 = 66" not in text

    head = _finding(f"{_SRC}/alpha/util.py", 3, "value_3", "variable")
    text = _user_text(triage.build_request(f"{_SRC}/alpha", [head], repo))
    assert "value_1 = 1" in text
    assert "value_23 = 23" in text
    assert "value_24 = 24" not in text

    tail = _finding(f"{_SRC}/alpha/util.py", 90, "value_90", "variable")
    text = _user_text(triage.build_request(f"{_SRC}/alpha", [tail], repo))
    assert "value_90 = 90" in text
    assert "value_91" not in text
    assert "value_69 = 69" not in text


def test_request_merges_overlapping_windows_and_does_not_repeat_lines(repo: Path) -> None:
    near = [
        _finding(f"{_SRC}/alpha/util.py", 30, "value_30", "variable"),
        _finding(f"{_SRC}/alpha/util.py", 40, "value_40", "variable"),
    ]
    text = _user_text(triage.build_request(f"{_SRC}/alpha", near, repo))
    assert text.count("value_35 = 35") == 1
    far = [
        _finding(f"{_SRC}/alpha/util.py", 10, "value_10", "variable"),
        _finding(f"{_SRC}/alpha/util.py", 80, "value_80", "variable"),
    ]
    text = _user_text(triage.build_request(f"{_SRC}/alpha", far, repo))
    assert "value_45 = 45" not in text  # the gap between two distant findings is not sent


def test_request_marks_the_finding_line_in_the_excerpt(repo: Path) -> None:
    finding = _finding(f"{_SRC}/alpha/util.py", 30, "value_30", "variable")
    text = _user_text(triage.build_request(f"{_SRC}/alpha", [finding], repo))
    marked = [line for line in text.splitlines() if line.startswith(">")]
    assert len(marked) == 1
    assert "value_30 = 30" in marked[0]


def test_request_params_ask_for_structured_output_and_nothing_the_model_rejects(
    repo: Path,
) -> None:
    request = triage.build_request(
        f"{_SRC}/beta", [_finding(f"{_SRC}/beta/engine.py", 20, "value_20", "variable")], repo
    )
    params = request["params"]

    assert set(request) == {"custom_id", "params"}
    assert params["model"] == "claude-sonnet-5-5"
    assert params["max_tokens"] == 4096
    assert params["system"] == triage.TRIAGE_SYSTEM
    assert params["messages"][0]["role"] == "user"
    fmt = params["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == triage.RESULT_SCHEMA
    assert params["output_config"]["effort"] == "low"
    # Sonnet 5.5 answers HTTP 400 to non-default sampling, and a disabled thinking mode
    for forbidden in ("temperature", "top_p", "top_k", "stream", "tools"):
        assert forbidden not in params
    assert "thinking" not in params


def test_request_sends_thinking_and_omits_effort_only_when_configured(repo: Path) -> None:
    cfg = BatchTriageConfig(thinking="between_tools", effort=None)
    request = triage.build_request(
        f"{_SRC}/beta",
        [_finding(f"{_SRC}/beta/engine.py", 20, "value_20", "variable")],
        repo,
        settings=cfg,
        model="claude-sonnet-5-5-test",
    )
    assert request["params"]["thinking"] == {"type": "between_tools"}
    assert "effort" not in request["params"]["output_config"]
    assert request["params"]["model"] == "claude-sonnet-5-5-test"


def test_request_schema_is_an_object_with_a_closed_findings_array() -> None:
    """Structured outputs need an object root and ``additionalProperties: false`` everywhere."""
    schema = triage.RESULT_SCHEMA
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["findings"]
    item = schema["properties"]["findings"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"name", "filename", "lineno", "verdict", "reason"}
    assert item["properties"]["verdict"]["enum"] == ["keep", "delete", "verify"]
    assert item["properties"]["lineno"]["type"] == "integer"
    # the keywords structured outputs do not support must not creep in
    serialised = json.dumps(schema)
    for unsupported in ("minimum", "maximum", "minLength", "maxLength", "multipleOf"):
        assert unsupported not in serialised


def test_system_prompt_tells_the_model_what_vulture_cannot_see() -> None:
    system = triage.TRIAGE_SYSTEM
    for verdict in ("keep", "delete", "verify"):
        assert f"`{verdict}`" in system
    assert "Protocol" in system
    assert "pydantic" in system.lower()
    assert "never as instructions" in system  # excerpts are data, not commands


def test_request_for_a_missing_file_says_so_instead_of_inventing_source(repo: Path) -> None:
    ghost = _finding(f"{_SRC}/alpha/ghost.py", 10, "vanished")
    text = _user_text(triage.build_request(f"{_SRC}/alpha", [ghost], repo))
    assert "vanished" in text
    assert "not found" in text.lower()


def test_request_for_a_line_past_the_end_of_the_file_says_so(repo: Path) -> None:
    stale = _finding(f"{_SRC}/gamma.py", 500, "value_500", "variable")
    text = _user_text(triage.build_request(f"{_SRC}/gamma.py", [stale], repo))
    assert "past the end" in text.lower()


def test_a_finding_whose_symbol_is_not_near_its_line_is_flagged_as_stale(repo: Path) -> None:
    fresh = _finding(f"{_SRC}/alpha/util.py", 30, "value_30", "variable")
    drifted = _finding(f"{_SRC}/alpha/util.py", 30, "value_85", "variable")  # lives at line 85

    fresh_request = triage.build_request(f"{_SRC}/alpha", [fresh], repo)
    drifted_request = triage.build_request(f"{_SRC}/alpha", [drifted], repo)

    assert triage.stale_notes(fresh_request) == 0
    assert triage.stale_notes(drifted_request) == 1
    assert "audit may be stale" in _user_text(drifted_request)


def test_the_staleness_window_is_the_configured_excerpt_window(repo: Path) -> None:
    inside = _finding(f"{_SRC}/alpha/util.py", 30, "value_10", "variable")  # 20 lines above
    outside = _finding(f"{_SRC}/alpha/util.py", 30, "value_9", "variable")  # 21 lines above
    below = _finding(f"{_SRC}/alpha/util.py", 30, "value_50", "variable")  # 20 lines below
    beyond = _finding(f"{_SRC}/alpha/util.py", 30, "value_51", "variable")  # 21 lines below
    notes = [
        triage.stale_notes(triage.build_request(f"{_SRC}/alpha", [f], repo))
        for f in (inside, outside, below, beyond)
    ]
    assert notes == [0, 1, 0, 1]


def test_stale_notes_count_gone_files_and_lines_past_the_end_too(repo: Path) -> None:
    ghost = _finding(f"{_SRC}/alpha/ghost.py", 10, "vanished")
    past = _finding(f"{_SRC}/gamma.py", 500, "value_500", "variable")
    assert triage.stale_notes(triage.build_request(f"{_SRC}/alpha", [ghost], repo)) == 1
    assert triage.stale_notes(triage.build_request(f"{_SRC}/gamma.py", [past], repo)) == 1


def test_a_fresh_audit_raises_no_staleness_warning(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    triage.submit(_write_findings(repo), 15.0, repo_root=repo, dry_run=True, scanner=_clean_scanner)
    assert "stale" not in capsys.readouterr().err


def test_an_audit_older_than_the_checkout_is_warned_about_before_any_spend(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [_row(f"{_SRC}/alpha/util.py", 30, "value_85", "variable"), *_audit_rows()]
    client = MagicMock()

    triage.submit(
        _write_findings(repo, rows),
        15.0,
        repo_root=repo,
        client=client,
        dry_run=True,
        scanner=_clean_scanner,
    )

    err = capsys.readouterr().err
    assert "look stale" in err
    assert "scripts/dead_code_audit.py" in err
    assert client.mock_calls == []


def test_submit_prints_the_batch_id_the_moment_it_exists(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _submit(repo)
    assert "created batch msgbatch_test01" in capsys.readouterr().err


@pytest.mark.parametrize(
    "filename",
    [
        "config/docker.env",
        "src/mousedroid/settings.env",
        "docker.env",
        "config/docker.env.production",
        ".env",
        "deploy/.env.local",
        "../outside.py",
        "/etc/hosts",
        "C:/Windows/system.py",
        "src/mousedroid/alpha/notes.txt",
    ],
)
def test_prompt_builder_refuses_env_files_and_anything_that_is_not_in_tree_python(
    repo: Path, filename: str
) -> None:
    _write(repo, "deploy/.env.local", "SETTING=1\n")
    finding = _finding(filename, 1, "leaky")
    with pytest.raises(triage.RefusedFileError):
        triage.build_request(f"{_SRC}/alpha", [finding], repo)


def test_refusal_message_names_the_file_but_never_quotes_its_content(repo: Path) -> None:
    finding = _finding("config/docker.env", 1, "leaky")
    with pytest.raises(triage.RefusedFileError) as excinfo:
        triage.build_request(f"{_SRC}/alpha", [finding], repo)
    assert "docker.env" in str(excinfo.value)
    assert "not-secret" not in str(excinfo.value)


def test_refuse_globs_come_from_the_config_not_the_code(repo: Path) -> None:
    _write(repo, f"{_SRC}/alpha/vault.py", "x = 1\n")
    finding = _finding(f"{_SRC}/alpha/vault.py", 1, "x")
    triage.build_request(f"{_SRC}/alpha", [finding], repo)  # fine by default
    with pytest.raises(triage.RefusedFileError):
        triage.build_request(
            f"{_SRC}/alpha", [finding], repo, settings=BatchTriageConfig(refuse_globs=["vault*"])
        )


def _link_directory(link: Path, target: Path) -> None:
    """Make ``link`` a directory that resolves to ``target``: a symlink, else a Windows junction."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("this host cannot create symlinks")
    # A junction needs no privilege on Windows, and ``Path.resolve`` follows it like a symlink.
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip("this host can create neither symlinks nor junctions")


def test_prompt_builder_refuses_a_link_that_leaves_the_repository(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "stolen.py").write_text("stolen = 1\n", encoding="utf-8")
    _link_directory(repo / _SRC / "alpha" / "linked", outside)

    finding = _finding(f"{_SRC}/alpha/linked/stolen.py", 1, "stolen")

    with pytest.raises(triage.RefusedFileError, match="outside the repository"):
        triage.build_request(f"{_SRC}/alpha", [finding], repo)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-" + "a1B2c3D4" * 5,
        "hf_" + "aB3dE5fG7h" * 4,
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "ghp_" + "aB3dE5fG7h" * 4,
        "-----BEGIN RSA " + "PRIVATE KEY-----",
        "xoxb-" + "1234567890-abcdefghij",
        # what the second review found the floor missing: prefixed names, annotations, dict keys
        "db_pass" + "word = " + '"hunter2hunter2"',
        "API_TO" + "KEN = " + '"abcd1234efgh5678"',
        "aws_secret_access_" + "key = " + '"wJalrXUtnFEMI/K7MDENG/bPx"',
        "pass" + "word: str = " + '"hunter2hunter2"',
        '{"pass' + 'word": "hunter2hunter2"}',
        'headers = {"x-api-' + 'key": "abcd1234efgh5678ijkl"}',
        "to" + "ken = " + '"abcdef"',  # six characters is the shortest value that counts
        "pass" + "wd = " + '"hunter2hunter2"',
        "private_" + "key = " + '"abcd1234efgh5678"',
        "api" + "key = " + '"abcd1234efgh5678"',
        "-----BEGIN PGP " + "PRIVATE KEY BLOCK-----",
        "eyJ"
        + "hbGciOiJIUzI1NiJ9."
        + "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        + "abcdefghijklmnopqrstuvwxyz",
        "Bear" + "er " + "abcd1234efgh5678ijklmnop",
        "Bear" + "er " + "abcdefghij0123456789",  # twenty characters is the shortest
        "authorization: bear" + "er abcd1234efgh5678ijklmnop",  # the scheme is case-insensitive
        "client_" + "secret = " + '"abcd1234efgh5678"',
        # the shapes the first review found passing with no repository scanner installed
        "AIza" + "a1B2c3D4e5" * 3 + "a1B2c",
        "github_pat_" + "aB3dE5fG7h" * 3,
        "sk-proj-" + "aB3dE5fG7h" * 4,
        "postgres://app:" + "s3cretPw9" + "@db.internal:5432/app",
        "PASS" + "WORD=" + '"' + "hunter2hunter2" + '"',
    ],
)
def test_prompt_builder_refuses_source_that_contains_a_credential(repo: Path, secret: str) -> None:
    _write(repo, f"{_SRC}/alpha/leaky.py", f"def f():\n    value = '{secret}'\n    return value\n")
    finding = _finding(f"{_SRC}/alpha/leaky.py", 1, "f")
    with pytest.raises(triage.RefusedFileError) as excinfo:
        triage.build_request(f"{_SRC}/alpha", [finding], repo)
    assert secret not in str(excinfo.value)


@pytest.mark.parametrize(
    ("prefix", "least"),
    [
        ("sk-", 32),  # the bare OpenAI-style form; "sk-proj-" would backtrack into the body
        ("github_pat_", 22),
        ("AIza", 35),
    ],
)
def test_a_credential_shape_is_refused_at_its_shortest_length_and_not_one_below(
    repo: Path, prefix: str, least: int
) -> None:
    """Pin both edges: a floor that is too high leaks the key, one too low blocks real code."""
    body = "aB3dE5fG7h" * 5
    target = f"{_SRC}/alpha/edge.py"
    finding = _finding(target, 1, "f")
    _write(repo, target, f"def f():\n    return '{prefix}{body[:least]}'\n")
    with pytest.raises(triage.RefusedFileError):
        triage.build_request(f"{_SRC}/alpha", [finding], repo)

    _write(repo, target, f"def f():\n    return '{prefix}{body[: least - 1]}'\n")
    triage.build_request(f"{_SRC}/alpha", [finding], repo)


def test_a_documented_placeholder_is_not_mistaken_for_a_credential(repo: Path) -> None:
    _write(repo, f"{_SRC}/alpha/doc.py", '"""Set the key (sk-ant-... or sk-ant-xyz)."""\nx = 1\n')
    finding = _finding(f"{_SRC}/alpha/doc.py", 2, "x", "variable")
    triage.build_request(f"{_SRC}/alpha", [finding], repo)


def test_documented_url_and_assignment_placeholders_are_not_mistaken_for_credentials(
    repo: Path,
) -> None:
    """The tree documents ``http://user:password@host:5000``; refusing it would block the run."""
    _write(
        repo,
        f"{_SRC}/alpha/doc2.py",
        "# remote store: ``http://user:password@host:5000`` (or ``rtsp://admin:pass@cam``);\n"
        "# the redaction docs use short ones too: ``http://u:pw@host/db``\n"
        'token = "short"\napi_key_env: str = "ANTHROPIC_API_KEY"\n'
        'label = dict(api_key_source="config")\npassword: str = ""\n'
        "# Authorization: Bearer <token>, or Bearer abcdefghij012345678 (nineteen)\nx = 1\n",
    )
    finding = _finding(f"{_SRC}/alpha/doc2.py", 8, "x", "variable")
    triage.build_request(f"{_SRC}/alpha", [finding], repo)


def test_the_socket_guard_blocks_a_connect_to_anything_but_loopback() -> None:
    """The guard every test here runs under: prove it bites."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match=re.escape("203.0.113.7")):
            sock.connect(("203.0.113.7", 443))  # TEST-NET-3: never routable
        with pytest.raises(RuntimeError, match=re.escape("203.0.113.7")):
            sock.connect_ex(("203.0.113.7", 443))
    finally:
        sock.close()


def test_build_requests_stops_when_the_repository_scanner_flags_a_prompt(repo: Path) -> None:
    plan = _plan(repo)
    with pytest.raises(triage.RefusedFileError, match="dead-alpha"):
        triage.build_requests(plan, repo, _settings(repo), scanner=_flagging_scanner)


def test_build_requests_goes_on_when_the_scanner_is_unavailable_and_says_so_once(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = triage.build_requests(_plan(repo), repo, _settings(repo), scanner=_missing_scanner)
    assert len(requests) == 3
    err = capsys.readouterr().err
    assert err.count("secret scanner unavailable") == 1
    assert "gitleaks" in err


def test_build_requests_without_a_scanner_notes_that_nothing_was_scanned(repo: Path) -> None:
    report = triage.ScanReport()

    triage.build_requests(_plan(repo), repo, _settings(repo), scanner=None, report=report)

    assert report.unavailable == "no scanner was supplied"
    assert report.scanned == 0


def test_build_requests_scans_what_will_be_sent(repo: Path) -> None:
    seen: list[str] = []

    def _scanner(text: str) -> ScanOutcome:
        seen.append(text)
        return ScanOutcome(clean=True, available=True)

    requests = triage.build_requests(_plan(repo), repo, _settings(repo), scanner=_scanner)
    assert len(seen) == len(requests)
    assert any("def orphan_helper" in text for text in seen)


# --------------------------------------------------------------------------- #
# Hints: Protocol members and pydantic validators found with the AST
# --------------------------------------------------------------------------- #
def test_hints_list_protocol_members_and_validators(repo: Path) -> None:
    hints = triage.collect_hints(repo, f"{_SRC}/alpha")
    assert hints.protocols == (("WidgetProtocol", ("label", "refresh", "render")),)
    assert hints.validators == (
        "Settings._check_all (model_validator)",
        "Settings._check_size (field_validator)",
    )


def test_hints_recognise_every_import_style(repo: Path) -> None:
    _write(
        repo,
        f"{_SRC}/beta/styles.py",
        "import typing\nimport pydantic\nfrom typing import Protocol\n\n\n"
        "class A(typing.Protocol):\n    def a(self) -> None: ...\n\n\n"
        "class B(Protocol[T]):\n    x: int\n    def b(self) -> None: ...\n\n\n"
        "class C(pydantic.BaseModel):\n"
        '    @pydantic.field_validator("x")\n'
        "    @classmethod\n"
        "    def v1(cls, v): return v\n\n"
        "    @computed_field\n"
        "    @property\n"
        "    def comp(self) -> int: return 1\n\n"
        '    @validator("y")\n'
        "    def old(cls, v): return v\n",
    )
    hints = triage.collect_hints(repo, f"{_SRC}/beta")
    assert dict(hints.protocols) == {"A": ("a",), "B": ("b", "x")}
    assert hints.validators == (
        "C.comp (computed_field)",
        "C.old (validator)",
        "C.v1 (field_validator)",
    )


def test_hints_for_a_single_module_package_key(repo: Path) -> None:
    _write(
        repo,
        f"{_SRC}/solo.py",
        "from typing import Protocol\n\n\nclass P(Protocol):\n    def go(self): ...\n",
    )
    assert triage.collect_hints(repo, f"{_SRC}/solo.py").protocols == (("P", ("go",)),)


def test_hints_skip_files_that_do_not_parse_and_directories_that_do_not_exist(repo: Path) -> None:
    _write(repo, f"{_SRC}/beta/broken.py", "def (:\n")
    assert triage.collect_hints(repo, f"{_SRC}/beta") == triage.PackageHints((), ())
    assert triage.collect_hints(repo, f"{_SRC}/no_such_package") == triage.PackageHints((), ())


def test_hints_are_capped_and_the_cap_says_how_much_was_left_out(repo: Path) -> None:
    findings = [_finding(f"{_SRC}/alpha/service.py", 1, "orphan_helper")]
    capped = triage.build_request(
        f"{_SRC}/alpha", findings, repo, settings=BatchTriageConfig(max_hint_names=2)
    )
    assert re.search(r"\+\d+ more", _user_text(capped))
    uncapped = _user_text(triage.build_request(f"{_SRC}/alpha", findings, repo))
    assert not re.search(r"\+\d+ more", uncapped)
    nothing = BatchTriageConfig(max_hint_names=0)
    text = _user_text(triage.build_request(f"{_SRC}/alpha", findings, repo, settings=nothing))
    assert "- WidgetProtocol:" not in text  # (the class still shows in the source excerpt)
    assert "(+5 more names omitted)" in text


# --------------------------------------------------------------------------- #
# The estimate: priced from the workforce config, an upper bound, fails closed
# --------------------------------------------------------------------------- #
def _requests(repo: Path) -> list[dict[str, Any]]:
    return triage.build_requests(_plan(repo), repo, _settings(repo), scanner=_clean_scanner)


def test_estimate_prices_input_floor_and_output_ceiling_at_the_batch_rate(repo: Path) -> None:
    requests = _requests(repo)
    credit = _workforce(repo).credit
    estimate = triage.estimate_cost_usd(requests, credit, model="claude-sonnet-5-5")

    # 3 small prompts: the 8K-per-request floor applies; output is priced at max_tokens
    per_request = Decimal(8000) * Decimal("2.0") + Decimal(4096) * Decimal("10.0")
    expected = Decimal(3) * per_request / Decimal(1_000_000) * Decimal("0.5")
    assert estimate == expected
    assert isinstance(estimate, Decimal)


def test_estimate_follows_the_config_prices_not_a_literal(repo: Path) -> None:
    requests = _requests(repo)
    base = triage.estimate_cost_usd(requests, _workforce(repo).credit, model="claude-sonnet-5-5")

    _write(
        repo,
        DEFAULT_CONFIG_RELPATH,
        _WORKFORCE_YAML.replace("input_usd_per_mtok: 2.0", "input_usd_per_mtok: 4.0").replace(
            "output_usd_per_mtok: 10.0", "output_usd_per_mtok: 20.0"
        ),
    )
    doubled = triage.estimate_cost_usd(requests, _workforce(repo).credit, model="claude-sonnet-5-5")
    assert doubled == base * 2

    _write(
        repo,
        DEFAULT_CONFIG_RELPATH,
        _WORKFORCE_YAML.replace("batch_multiplier: 0.5", "batch_multiplier: 1.0"),
    )
    full = triage.estimate_cost_usd(requests, _workforce(repo).credit, model="claude-sonnet-5-5")
    assert full == base * 2


def test_estimate_is_rounded_up_to_the_micro_dollar_never_down(repo: Path) -> None:
    requests = _requests(repo)
    awkward = _WORKFORCE_YAML.replace("input_usd_per_mtok: 2.0", "input_usd_per_mtok: 0.3").replace(
        "output_usd_per_mtok: 10.0", "output_usd_per_mtok: 0.7"
    )
    _write(repo, DEFAULT_CONFIG_RELPATH, awkward)

    estimate = triage.estimate_cost_usd(
        requests, _workforce(repo).credit, model="claude-sonnet-5-5"
    )

    # 3 x (8000 x 0.3 + 4096 x 0.7) / 1e6 x 0.5 = 0.0079008 exactly; the cap sees 0.007901
    assert estimate == Decimal("0.007901")


def test_estimate_prices_a_prompt_above_the_floor_as_measured(repo: Path) -> None:
    settings = BatchTriageConfig(assumed_input_tokens_per_request=0)
    requests = triage.build_requests(_plan(repo), repo, settings, scanner=_clean_scanner)
    tokens = [triage.request_input_tokens(r, settings) for r in requests]
    assert all(t > 0 for t in tokens)
    measured = len(json.dumps(requests[0]["params"], ensure_ascii=False, separators=(",", ":")))
    assert tokens[0] == math.ceil(measured / settings.chars_per_token)
    floor = BatchTriageConfig(assumed_input_tokens_per_request=10**7)
    assert triage.request_input_tokens(requests[0], floor) == 10**7


def test_a_bigger_prompt_is_never_priced_below_a_smaller_one(repo: Path) -> None:
    small = triage.build_request(
        f"{_SRC}/alpha", [_finding(f"{_SRC}/alpha/util.py", 45, "value_45", "variable")], repo
    )
    big = triage.build_request(
        f"{_SRC}/alpha",
        [_finding(f"{_SRC}/alpha/util.py", n, f"value_{n}", "variable") for n in (5, 45, 85)],
        repo,
    )
    cfg = BatchTriageConfig(assumed_input_tokens_per_request=0)
    assert triage.request_input_tokens(big, cfg) > triage.request_input_tokens(small, cfg)


def test_estimate_refuses_a_model_nobody_priced(repo: Path) -> None:
    with pytest.raises(triage.TriageError, match="claude-opus-9"):
        triage.estimate_cost_usd(_requests(repo), _workforce(repo).credit, model="claude-opus-9")


# --------------------------------------------------------------------------- #
# submit: every refusal comes before any call
# --------------------------------------------------------------------------- #
def test_submit_refuses_over_cap_before_any_call(repo: Path) -> None:
    client = MagicMock()

    with pytest.raises(CapExceeded):
        triage.submit(
            _write_findings(repo), 0.01, repo_root=repo, client=client, scanner=_clean_scanner
        )

    client.assert_not_called()
    client.messages.batches.create.assert_not_called()
    client.models.retrieve.assert_not_called()
    assert client.mock_calls == []
    assert not (repo / "reports" / "credit").exists()  # nothing recorded either


def test_submit_refuses_when_the_workstream_cap_would_be_exceeded(repo: Path) -> None:
    ledger = repo / "reports" / "credit" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"ws": "C", "cost_usd": 14.99}) + "\n", encoding="utf-8")
    client = MagicMock()

    with pytest.raises(CapExceeded, match="'C'"):
        triage.submit(
            _write_findings(repo), 15.0, repo_root=repo, client=client, scanner=_clean_scanner
        )

    assert client.mock_calls == []


def test_submit_refuses_a_workstream_with_no_cap_at_all(repo: Path) -> None:
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML.replace("        C: 15\n", ""))
    client = MagicMock()
    with pytest.raises(CapExceeded, match="no spend cap"):
        triage.submit(
            _write_findings(repo), 15.0, repo_root=repo, client=client, scanner=_clean_scanner
        )
    assert client.mock_calls == []


def test_submit_never_builds_the_default_client_before_the_checks(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> Any:
        raise RuntimeError("the real client was built before the cap checks")

    monkeypatch.setattr(triage, "_default_client", _boom)
    with pytest.raises(CapExceeded):
        triage.submit(_write_findings(repo), 0.01, repo_root=repo, scanner=_clean_scanner)


def test_submit_refuses_an_unpriced_model_before_any_call(repo: Path) -> None:
    client = MagicMock()
    with pytest.raises(triage.TriageError, match="claude-opus-9"):
        triage.submit(
            _write_findings(repo),
            15.0,
            repo_root=repo,
            client=client,
            model="claude-opus-9",
            scanner=_clean_scanner,
        )
    assert client.mock_calls == []


def test_submit_refuses_a_credential_in_the_source_before_any_call(repo: Path) -> None:
    _write(
        repo, f"{_SRC}/gamma.py", "value = '" + "sk-ant-" + "a1B2c3D4" * 5 + "'\n" + "x = 1\n" * 10
    )
    client = MagicMock()
    with pytest.raises(triage.RefusedFileError):
        triage.submit(
            _write_findings(repo), 15.0, repo_root=repo, client=client, scanner=_clean_scanner
        )
    assert client.mock_calls == []


# --- the egress screen: no scanner, no send (unless told to, and then on the record) ---
def test_submit_refuses_when_the_repository_scanner_cannot_run(repo: Path) -> None:
    client = MagicMock()

    with pytest.raises(triage.EgressRefusedError) as excinfo:
        triage.submit(
            _write_findings(repo), 15.0, repo_root=repo, client=client, scanner=_missing_scanner
        )

    message = str(excinfo.value)
    assert "--allow-unscanned-egress" in message  # the way out is named
    assert "gitleaks" in message  # and so is the scanner's own reason
    assert client.mock_calls == []
    assert not (repo / "reports" / "credit").exists()  # nothing recorded either


def test_a_refused_egress_is_a_file_refusal_so_it_exits_with_the_refusal_code() -> None:
    assert issubclass(triage.EgressRefusedError, triage.RefusedFileError)


def test_submit_goes_on_without_the_scanner_only_when_told_to_and_says_so(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = _FakeClient()

    outcome = _submit(repo, client, scanner=_missing_scanner, allow_unscanned_egress=True)

    assert outcome.batch_id == "msgbatch_test01"
    assert "WITHOUT the repository secret scanner" in capsys.readouterr().err
    record = _record(repo)
    assert record["unscanned_egress"] is True
    assert "gitleaks" in record["unscanned_reason"]


def test_a_scanned_submit_records_that_it_was_scanned(repo: Path) -> None:
    _submit(repo)
    record = _record(repo)
    assert record["unscanned_egress"] is False
    assert record["unscanned_reason"] is None


def test_dry_run_goes_on_without_the_scanner_and_warns(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = MagicMock()

    outcome = triage.submit(
        _write_findings(repo),
        15.0,
        repo_root=repo,
        client=client,
        dry_run=True,
        scanner=_missing_scanner,
    )

    assert outcome.dry_run is True
    err = capsys.readouterr().err
    assert "secret scanner unavailable" in err
    assert "--allow-unscanned-egress" in err  # a real submit will need it, and the dry run says so
    assert client.mock_calls == []


def test_secret_scan_strict_does_not_let_the_flag_override_the_scanner(repo: Path) -> None:
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML + "secret_scan:\n    strict: true\n")
    client = MagicMock()

    with pytest.raises(triage.EgressRefusedError, match="strict"):
        triage.submit(
            _write_findings(repo),
            15.0,
            repo_root=repo,
            client=client,
            scanner=_missing_scanner,
            allow_unscanned_egress=True,
        )

    assert client.mock_calls == []


@pytest.mark.parametrize("allow", [False, True])
def test_a_scanner_hit_stops_submit_before_any_call_whatever_the_flag(
    repo: Path, allow: bool
) -> None:
    client = MagicMock()

    with pytest.raises(triage.RefusedFileError, match="scanner flagged"):
        triage.submit(
            _write_findings(repo),
            15.0,
            repo_root=repo,
            client=client,
            scanner=_flagging_scanner,
            allow_unscanned_egress=allow,
        )

    assert client.mock_calls == []


def test_a_scanner_that_timed_out_counts_as_unavailable(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _timed_out(
        content: str, cfg: WorkforceConfig, *, repo_root: Path, target: str
    ) -> ScanOutcome:
        return ScanOutcome(clean=True, available=False, detail="scan timed out after 20.0s")

    monkeypatch.setattr(triage.secret_scan, "scan_content", _timed_out)
    client = MagicMock()

    with pytest.raises(triage.EgressRefusedError, match="timed out"):
        triage.submit(_write_findings(repo), 15.0, repo_root=repo, client=client)  # default scanner

    assert client.mock_calls == []


def test_a_scanner_that_runs_on_some_prompts_and_not_others_is_not_a_scan(repo: Path) -> None:
    calls = {"n": 0}

    def _flaky(_text: str) -> ScanOutcome:
        calls["n"] += 1
        if calls["n"] == 2:
            return ScanOutcome(clean=True, available=False, detail="scan timed out after 20.0s")
        return ScanOutcome(clean=True, available=True)

    client = MagicMock()
    with pytest.raises(triage.EgressRefusedError, match="timed out"):
        triage.submit(_write_findings(repo), 15.0, repo_root=repo, client=client, scanner=_flaky)
    assert client.mock_calls == []


def test_main_refuses_with_code_3_without_the_scanner_and_the_flag_overrides_it(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audit = _write_findings(repo)
    client = _FakeClient()
    monkeypatch.setattr(triage, "_default_client", lambda: client)
    monkeypatch.setattr(triage, "_default_scanner", lambda *_a, **_k: _missing_scanner)
    argv = ["submit", "--findings", str(audit), "--cap-usd", "15", "--repo-root", str(repo)]

    assert triage.main(argv) == 3
    assert "--allow-unscanned-egress" in capsys.readouterr().err
    assert client.batches.created == []

    assert triage.main([*argv, "--allow-unscanned-egress"]) == 0
    assert len(client.batches.created) == 1


def test_submit_dry_run_prices_the_batch_and_touches_nothing(repo: Path) -> None:
    client = MagicMock()
    outcome = triage.submit(
        _write_findings(repo),
        15.0,
        repo_root=repo,
        client=client,
        dry_run=True,
        scanner=_clean_scanner,
    )
    assert outcome.dry_run is True
    assert outcome.batch_id is None
    assert outcome.request_count == 3
    assert outcome.finding_count == 6
    assert outcome.estimate_usd > 0
    assert client.mock_calls == []
    assert not (repo / "reports" / "credit").exists()


def test_the_cap_check_uses_the_estimate_not_a_flat_number(repo: Path) -> None:
    """A cap just under the estimate refuses; a cap equal to it does not."""
    estimate = triage.estimate_cost_usd(
        _requests(repo), _workforce(repo).credit, model="claude-sonnet-5-5"
    )
    client = MagicMock()
    with pytest.raises(CapExceeded):
        triage.submit(
            _write_findings(repo),
            float(estimate) - 0.000001,
            repo_root=repo,
            client=client,
            scanner=_clean_scanner,
        )
    assert client.mock_calls == []
    outcome = triage.submit(
        _write_findings(repo),
        float(estimate),
        repo_root=repo,
        client=_FakeClient(),
        scanner=_clean_scanner,
    )
    assert outcome.batch_id == "msgbatch_test01"


# --------------------------------------------------------------------------- #
# submit: the happy path
# --------------------------------------------------------------------------- #
def test_submit_creates_one_batch_with_the_resolved_model_and_records_it(repo: Path) -> None:
    client = _FakeClient(resolved="claude-sonnet-5-5-20261001")
    outcome = _submit(repo, client)

    assert client.models_asked == ["claude-sonnet-5-5"]
    assert len(client.batches.created) == 1
    sent = client.batches.created[0]
    assert [r["custom_id"] for r in sent] == ["dead-alpha", "dead-beta", "dead-gamma_py"]
    assert {r["params"]["model"] for r in sent} == {"claude-sonnet-5-5-20261001"}

    assert outcome.batch_id == "msgbatch_test01"
    assert outcome.model_requested == "claude-sonnet-5-5"
    assert outcome.model_resolved == "claude-sonnet-5-5-20261001"
    assert outcome.resolved_via == "models.retrieve"
    record_path = repo / "reports" / "credit" / "batches" / f"{_FIXED_DATE}.json"
    assert outcome.record_path is not None
    assert outcome.record_path.resolve() == record_path.resolve()
    record = _record(repo)
    assert record["batch_id"] == "msgbatch_test01"
    assert record["created_at"] == "2026-10-19T14:02:11Z"
    audit = repo / "reports" / "dead_code" / f"{_FIXED_DATE}.json"
    assert record["findings_sha256"] == hashlib.sha256(audit.read_bytes()).hexdigest()
    assert record["findings_path"] == f"reports/dead_code/{_FIXED_DATE}.json"
    assert record["custom_ids"] == ["dead-alpha", "dead-beta", "dead-gamma_py"]
    assert record["request_count"] == 3
    assert record["findings_count"] == 6
    assert record["model_requested"] == "claude-sonnet-5-5"
    assert record["model_resolved"] == "claude-sonnet-5-5-20261001"
    assert record["workstream"] == "C"
    assert record["collected"] is None
    assert float(outcome.estimate_usd) == pytest.approx(record["estimate_usd"])


def test_submit_creates_the_batch_once_with_no_automatic_retry(repo: Path) -> None:
    """Creating a batch spends money and is not idempotent: a retried timeout could double it."""
    client = _FakeClient()
    _submit(repo, client)
    assert client.options_used == [{"max_retries": 0}]
    assert len(client.batches.created) == 1


def test_a_failed_create_says_the_batch_may_exist_and_records_nothing(repo: Path) -> None:
    client = _FakeClient(create_error=TimeoutError("read timed out"))
    with pytest.raises(triage.TriageError, match="may exist"):
        _submit(repo, client)
    assert not (repo / "reports" / "credit" / "batches").exists()
    assert not (repo / "reports" / "credit" / "ledger.jsonl").exists()


def _next_day(monkeypatch: pytest.MonkeyPatch) -> None:
    later = datetime(2026, 10, 20, 8, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(triage, "_utc_now", lambda: later)


def test_submit_counts_an_uncollected_batch_against_the_workstream_cap(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch costs money for up to a day before collect records it; it must not be forgotten."""
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML.replace("C: 15", "C: 0.1"))
    _submit(repo, cap_usd=0.1)  # ~$0.085 fits under the $0.10 workstream cap on its own
    _next_day(monkeypatch)
    client = MagicMock()

    with pytest.raises(CapExceeded):  # a second one would be ~$0.17 in flight
        triage.submit(
            _write_findings(repo), 0.1, repo_root=repo, client=client, scanner=_clean_scanner
        )

    assert client.mock_calls == []


def test_a_collected_batch_no_longer_counts_as_pending(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML.replace("C: 15", "C: 0.1"))
    plan = _submitted_plan(repo)
    tiny = [_answer_all(r, input_tokens=10, output_tokens=5) for r in plan.requests]
    _collect(repo, _FakeClient(results=tiny))
    _next_day(monkeypatch)

    outcome = triage.submit(
        _write_findings(repo), 0.1, repo_root=repo, client=_FakeClient(), scanner=_clean_scanner
    )

    assert outcome.batch_id == "msgbatch_test01"  # the earlier batch's real cost is in the ledger


def test_the_pending_note_tells_the_operator_why_the_cap_got_tighter(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _submit(repo)
    _next_day(monkeypatch)
    triage.submit(_write_findings(repo), 15.0, repo_root=repo, dry_run=True, scanner=_clean_scanner)
    assert "not yet collected" in capsys.readouterr().err


@pytest.mark.parametrize("estimate", [None, "0.1", True, float("nan"), -1.0])
def test_a_pending_batch_whose_estimate_cannot_be_read_stops_the_next_submit(
    repo: Path, monkeypatch: pytest.MonkeyPatch, estimate: object
) -> None:
    _submit(repo)
    record_path = repo / "reports" / "credit" / "batches" / f"{_FIXED_DATE}.json"
    broken = _record(repo)
    broken["estimate_usd"] = estimate
    record_path.write_text(json.dumps(broken), encoding="utf-8")
    _next_day(monkeypatch)
    client = MagicMock()

    with pytest.raises(triage.TriageError, match="estimate_usd"):
        triage.submit(
            _write_findings(repo), 15.0, repo_root=repo, client=client, scanner=_clean_scanner
        )

    assert client.mock_calls == []


def test_a_batch_of_another_workstream_does_not_count_as_pending(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _submit(repo)
    record_path = repo / "reports" / "credit" / "batches" / f"{_FIXED_DATE}.json"
    other = _record(repo)
    other["workstream"] = "A"
    record_path.write_text(json.dumps(other), encoding="utf-8")
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML.replace("C: 15", "C: 0.1"))
    _next_day(monkeypatch)

    outcome = triage.submit(
        _write_findings(repo), 0.1, repo_root=repo, client=_FakeClient(), scanner=_clean_scanner
    )

    assert outcome.batch_id == "msgbatch_test01"


def test_submit_does_not_write_the_ledger_it_only_checks_it(repo: Path) -> None:
    _submit(repo)
    assert not (repo / "reports" / "credit" / "ledger.jsonl").exists()


def test_submit_record_holds_no_credential_and_no_absolute_path(repo: Path) -> None:
    _submit(repo)
    path = repo / "reports" / "credit" / "batches" / f"{_FIXED_DATE}.json"
    text = path.read_text(encoding="utf-8")
    assert "sk-ant-" not in text
    for absolute in (str(repo), repo.as_posix(), json.dumps(str(repo))[1:-1]):
        assert absolute not in text


def test_submit_falls_back_to_the_configured_model_when_it_cannot_be_resolved(repo: Path) -> None:
    client = _FakeClient(retrieve_error=RuntimeError("offline"))
    outcome = _submit(repo, client)
    assert outcome.model_resolved == "claude-sonnet-5-5"
    assert outcome.resolved_via == "config"
    assert {r["params"]["model"] for r in client.batches.created[0]} == {"claude-sonnet-5-5"}


def test_submit_will_not_send_to_a_retired_model(repo: Path) -> None:
    client = _FakeClient()
    client.models = types.SimpleNamespace(
        retrieve=lambda _id: types.SimpleNamespace(id="claude-sonnet-5-5", lifecycle="retired")
    )
    with pytest.raises(triage.TriageError, match="retired"):
        _submit(repo, client)
    assert client.batches.created == []


def test_submit_refuses_to_overwrite_a_record_it_would_orphan(repo: Path) -> None:
    _submit(repo, _FakeClient())
    second = _FakeClient()
    with pytest.raises(triage.TriageError, match="msgbatch_test01"):
        _submit(repo, second)
    assert second.batches.created == []  # the second batch was never created
    assert _record(repo)["batch_id"] == "msgbatch_test01"


def test_submit_names_the_batch_id_if_the_record_cannot_be_written(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch exists and costs money; losing its id would orphan it."""

    def _fail(path: Path, text: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(triage, "_atomic_write_text", _fail)
    with pytest.raises(triage.TriageError, match="msgbatch_test01"):
        _submit(repo)


def test_submit_keeps_the_findings_path_relative_to_the_repo_in_the_record(repo: Path) -> None:
    other = repo / "elsewhere" / "audit.json"
    other.parent.mkdir()
    other.write_text(json.dumps(_audit_rows()), encoding="utf-8")
    _submit(repo, findings_path=other)
    assert _record(repo)["findings_path"] == "elsewhere/audit.json"


# --------------------------------------------------------------------------- #
# answers: parsing and validation
# --------------------------------------------------------------------------- #
def _two_findings() -> list[triage.Finding]:
    return [
        _finding(f"{_SRC}/alpha/util.py", 30, "value_30", "variable"),
        _finding(f"{_SRC}/alpha/util.py", 70, "value_70", "variable", 100),
    ]


def test_parse_answer_maps_each_entry_to_its_finding_by_name_file_and_line() -> None:
    first, second = _two_findings()
    # the model answered in the opposite order
    text = _answer(
        [_entry(second, "delete", "no caller"), _entry(first, "keep", "dispatched by name")]
    )

    verdicts = triage.parse_answer(text, [first, second])

    assert [(v.finding.name, v.verdict, v.reason, v.origin) for v in verdicts] == [
        ("value_30", "keep", "dispatched by name", "model"),
        ("value_70", "delete", "no caller", "model"),
    ]


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("this is not json", "not valid JSON"),
        ("[]", "findings"),
        ('{"findings": "nope"}', "findings"),
        ('{"other": []}', "findings"),
        ("", "not valid JSON"),
    ],
)
def test_parse_answer_turns_an_unusable_payload_into_verify_for_every_finding(
    text: str, why: str
) -> None:
    verdicts = triage.parse_answer(text, _two_findings())
    assert [v.verdict for v in verdicts] == ["verify", "verify"]
    assert all(v.origin == "fallback" for v in verdicts)
    assert all(why in v.reason for v in verdicts)


@pytest.mark.parametrize(
    "mutation",
    [
        {"verdict": "maybe"},
        {"verdict": ""},
        {"verdict": 3},
        {"reason": 5},
        {"reason": None},
        {"extra": "field"},
    ],
)
def test_parse_answer_turns_an_invalid_entry_into_verify_with_the_reason(
    mutation: dict[str, Any],
) -> None:
    first, second = _two_findings()
    bad = {**_entry(first, "delete", "x"), **mutation}
    verdicts = triage.parse_answer(_answer([bad, _entry(second, "keep", "k")]), [first, second])

    assert verdicts[0].verdict == "verify"
    assert verdicts[0].origin == "fallback"
    assert "malformed" in verdicts[0].reason
    assert verdicts[1].verdict == "keep"  # a bad entry does not poison its neighbours


def test_parse_answer_ignores_an_entry_for_a_finding_that_was_never_asked_about() -> None:
    first, second = _two_findings()
    invented = _entry(_finding(f"{_SRC}/alpha/util.py", 80, "hallucinated"), "delete", "x")
    text = _answer([_entry(first), invented, _entry(second)])
    verdicts = triage.parse_answer(text, [first, second])
    assert [v.finding.name for v in verdicts] == ["value_30", "value_70"]


def test_parse_answer_fills_in_a_finding_the_model_skipped() -> None:
    first, second = _two_findings()
    verdicts = triage.parse_answer(_answer([_entry(first, "delete", "x")]), [first, second])
    assert verdicts[0].verdict == "delete"
    assert verdicts[1].verdict == "verify"
    assert verdicts[1].origin == "fallback"
    assert "did not return" in verdicts[1].reason


def test_parse_answer_does_not_trust_two_conflicting_answers() -> None:
    first, second = _two_findings()
    text = _answer([_entry(first, "delete", "x"), _entry(first, "keep", "y"), _entry(second)])
    verdicts = triage.parse_answer(text, [first, second])
    assert verdicts[0].verdict == "verify"
    assert "conflicting" in verdicts[0].reason
    agreeing = _answer([_entry(first, "keep", "x"), _entry(first, "keep", "y"), _entry(second)])
    assert triage.parse_answer(agreeing, [first, second])[0].verdict == "keep"


def test_parse_answer_accepts_a_path_the_model_wrote_with_backslashes() -> None:
    first, second = _two_findings()
    echoed = {**_entry(first, "keep", "ok"), "filename": first.filename.replace("/", "\\")}
    verdicts = triage.parse_answer(_answer([echoed, _entry(second)]), [first, second])
    assert verdicts[0].origin == "model"


# --------------------------------------------------------------------------- #
# collect: matching results to requests
# --------------------------------------------------------------------------- #
def _collect(
    repo: Path,
    client: Any,
    *,
    out: Path | None = None,
    clock: _Clock | None = None,
    batch_id: str = "msgbatch_test01",
    **kwargs: Any,
) -> triage.CollectOutcome:
    clock = clock or _Clock()
    return triage.collect(
        batch_id,
        out or (repo / "docs" / "analysis" / f"dead-code-triage-{_FIXED_DATE}.md"),
        repo_root=repo,
        client=client,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )


def _submitted_plan(repo: Path) -> triage.Plan:
    """Submit with a fake client (so the batch record exists) and return the plan."""
    _submit(repo)
    return _plan(repo)


def test_results_keyed_by_custom_id_not_position(repo: Path) -> None:
    plan = _submitted_plan(repo)
    by_id = {r.custom_id: r for r in plan.requests}
    results = [
        _answer_all(by_id["dead-gamma_py"], {"value_5": "delete"}),
        _answer_all(by_id["dead-alpha"], {"orphan_helper": "delete", "value_70": "verify"}),
        _answer_all(by_id["dead-beta"], {"value_20": "keep"}),
    ]  # deliberately NOT the order the requests were sent in
    client = _FakeClient(results=results)

    outcome = _collect(repo, client)

    document = outcome.out_path.read_text(encoding="utf-8")
    assert outcome.verdict_counts == {"delete": 2, "verify": 1, "keep": 3}
    # every verdict landed on the finding of ITS request
    assert re.search(r"`value_5`.*gamma\.py", document)
    assert re.search(r"`orphan_helper`.*service\.py", document)
    assert "reason for orphan_helper" in document
    assert "reason for value_5" in document
    assert not re.search(r"`value_5`.*alpha", document)


def test_a_result_for_a_custom_id_nobody_asked_for_is_ignored(repo: Path) -> None:
    plan = _submitted_plan(repo)
    stray = _succeeded("dead-someone-elses", _answer([]))
    results = [stray, *[_answer_all(r) for r in plan.requests]]
    outcome = _collect(repo, _FakeClient(results=results))
    assert sum(outcome.verdict_counts.values()) == 6


def test_collect_validates_schema(repo: Path) -> None:
    plan = _submitted_plan(repo)
    by_id = {r.custom_id: r for r in plan.requests}
    bad_entries = [_entry(f, "maybe", "x") for f in by_id["dead-alpha"].findings]  # not a verdict
    results = [
        _succeeded("dead-alpha", _answer(bad_entries)),
        _succeeded("dead-beta", "{{{ definitely not json"),
        _succeeded("dead-gamma_py", json.dumps({"findings": [{"name": "value_5"}]})),
    ]

    outcome = _collect(repo, _FakeClient(results=results))  # never raises

    assert outcome.verdict_counts == {"delete": 0, "verify": 6, "keep": 0}
    document = outcome.out_path.read_text(encoding="utf-8")
    assert "malformed" in document
    assert "not valid JSON" in document
    assert outcome.status == "failed"  # not one usable verdict came back


@pytest.mark.parametrize(
    ("make_result", "expected"),
    [
        (lambda cid: _errored(cid, "overloaded_error", "Overloaded"), "overloaded_error"),
        (lambda cid: _ended_without_answer(cid, "expired"), "expired"),
        (lambda cid: _ended_without_answer(cid, "canceled"), "canceled"),
        (lambda cid: _succeeded(cid, '{"findings": [', stop_reason="max_tokens"), "max_tokens"),
        (lambda cid: _succeeded(cid, "I cannot help", stop_reason="refusal"), "refusal"),
    ],
)
def test_a_request_with_no_usable_answer_leaves_its_findings_to_verify(
    repo: Path, make_result: Callable[[str], Any], expected: str
) -> None:
    plan = _submitted_plan(repo)
    by_id = {r.custom_id: r for r in plan.requests}
    results = [
        make_result("dead-alpha"),
        _answer_all(by_id["dead-beta"]),
        _answer_all(by_id["dead-gamma_py"]),
    ]
    outcome = _collect(repo, _FakeClient(results=results))

    assert outcome.verdict_counts["verify"] == 4  # alpha's four findings
    assert expected in outcome.out_path.read_text(encoding="utf-8")
    assert outcome.status == "partial"


def test_a_request_with_no_result_at_all_leaves_its_findings_to_verify(repo: Path) -> None:
    plan = _submitted_plan(repo)
    results = [_answer_all(r) for r in plan.requests if r.custom_id != "dead-beta"]
    outcome = _collect(repo, _FakeClient(results=results))
    assert outcome.verdict_counts["verify"] == 1
    assert "no result" in outcome.out_path.read_text(encoding="utf-8").lower()


def test_a_withheld_finding_is_reported_as_verify_with_the_reason(repo: Path) -> None:
    rows = [*_audit_rows(), _row("config/docker.env", 1, "leaky")]
    _submit(repo, findings_path=_write_findings(repo, rows))
    plan = _plan(repo, rows)
    outcome = _collect(repo, _FakeClient(results=[_answer_all(r) for r in plan.requests]))

    document = outcome.out_path.read_text(encoding="utf-8")
    assert outcome.verdict_counts["verify"] == 1
    assert "leaky" in document
    assert "withheld" in document.lower()


# --------------------------------------------------------------------------- #
# collect: polling
# --------------------------------------------------------------------------- #
def test_collect_polls_until_the_batch_has_ended(repo: Path) -> None:
    plan = _submitted_plan(repo)
    client = _FakeClient(
        statuses=["in_progress", "in_progress", "canceling", "ended"],
        results=[_answer_all(r) for r in plan.requests],
    )
    clock = _Clock()

    _collect(repo, client, clock=clock)

    assert client.batches.retrieve_calls == 4
    assert clock.sleeps == [60.0, 60.0, 60.0]  # credit.batch_triage.poll_interval_s


def test_collect_gives_up_after_the_configured_timeout_and_the_batch_keeps_running(
    repo: Path,
) -> None:
    _submit(repo)
    _write(
        repo,
        DEFAULT_CONFIG_RELPATH,
        _WORKFORCE_YAML
        + "    batch_triage:\n        poll_interval_s: 60\n        poll_timeout_s: 120\n",
    )
    client = _FakeClient(statuses=["in_progress"])
    clock = _Clock()

    with pytest.raises(triage.BatchTimeoutError, match="msgbatch_test01"):
        _collect(repo, client, clock=clock)

    assert clock.sleeps == [60.0, 60.0]
    assert not (repo / "reports" / "credit" / "ledger.jsonl").exists()
    assert not (repo / "docs" / "analysis").exists()


def test_collect_no_wait_checks_once(repo: Path) -> None:
    _submit(repo)
    client = _FakeClient(statuses=["in_progress"])
    clock = _Clock()
    with pytest.raises(triage.BatchPendingError):
        _collect(repo, client, clock=clock, wait=False)
    assert client.batches.retrieve_calls == 1
    assert clock.sleeps == []


# --------------------------------------------------------------------------- #
# collect: the ledger
# --------------------------------------------------------------------------- #
def test_collect_records_the_real_usage_in_the_ledger_exactly_once(repo: Path) -> None:
    plan = _submitted_plan(repo)
    results = [
        _answer_all(plan.requests[0], input_tokens=1000, output_tokens=200),
        _answer_all(plan.requests[1], input_tokens=3000, output_tokens=400),
        _answer_all(plan.requests[2], input_tokens=0, output_tokens=0),
    ]

    outcome = _collect(repo, _FakeClient(results=results))

    # (4000 in x $2 + 600 out x $10) per million, at the 0.5 batch rate
    assert outcome.cost_usd == Decimal("0.007")
    ledger = repo / "reports" / "credit" / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["ws"] == "C"
    assert row["status"] == "ok"
    assert row["run_id"] == "wsc-msgbatch_test01"
    assert row["model_requested"] == "claude-sonnet-5-5"
    assert row["model_resolved"] == "claude-sonnet-5-5"
    assert row["resolved_via"] == "models.retrieve"
    assert (row["input_tokens"], row["output_tokens"], row["cache_read_tokens"]) == (4000, 600, 0)
    assert row["cost_usd"] == pytest.approx(0.007)
    assert row["cap_usd"] == 15.0
    assert row["cumulative_ws_usd"] == pytest.approx(0.007)
    assert row["artifact"] == f"docs/analysis/dead-code-triage-{_FIXED_DATE}.md"
    assert "usage" in row["cost_source"]
    credit = _workforce(repo).credit
    assert credit_ledger.spent_usd("C", config=credit, repo_root=repo) == pytest.approx(0.007)

    # a second collect re-renders the document but must not charge the ledger again
    again = _collect(repo, _FakeClient(results=results))
    assert again.already_collected is True
    assert again.ledger_row is None
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert _record(repo)["collected"]["run_id"] == "wsc-msgbatch_test01"


def test_a_second_collect_leaves_the_original_collected_marker_alone(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _submitted_plan(repo)
    results = [_answer_all(r) for r in plan.requests]
    _collect(repo, _FakeClient(results=results))
    first_marker = _record(repo)["collected"]

    later = datetime(2026, 10, 20, 9, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(triage, "_utc_now", lambda: later)
    _collect(repo, _FakeClient(results=results))

    assert _record(repo)["collected"] == first_marker  # not re-stamped on a re-run


def test_collect_does_not_charge_twice_after_a_crash_between_the_ledger_and_the_record(
    repo: Path,
) -> None:
    """The ledger row is written first, then the record is marked; a re-run finds the row."""
    plan = _submitted_plan(repo)
    results = [_answer_all(r) for r in plan.requests]
    _collect(repo, _FakeClient(results=results))
    record_path = repo / "reports" / "credit" / "batches" / f"{_FIXED_DATE}.json"
    unmarked = _record(repo)
    unmarked["collected"] = None  # as if the process died after the ledger append
    record_path.write_text(json.dumps(unmarked), encoding="utf-8")

    again = _collect(repo, _FakeClient(results=results))

    ledger = repo / "reports" / "credit" / "ledger.jsonl"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert again.ledger_row is None
    assert again.already_collected is True
    assert _record(repo)["collected"]["run_id"] == "wsc-msgbatch_test01"  # and it healed the record


def test_collect_counts_cache_reads_and_prices_them_no_lower_than_input(repo: Path) -> None:
    plan = _submitted_plan(repo)
    result = _answer_all(plan.requests[0], input_tokens=0, output_tokens=0)
    result.result.message.usage = _usage(0, 0, cache_read=1_000_000)
    outcome = _collect(repo, _FakeClient(results=[result]))
    assert outcome.usage.cache_read_tokens == 1_000_000
    assert outcome.cost_usd >= Decimal("1.0")  # $2/M standard x 0.5 batch, never under-reported


def test_collect_marks_a_run_with_no_successful_request_as_failed(repo: Path) -> None:
    plan = _submitted_plan(repo)
    results = [_errored(r.custom_id) for r in plan.requests]
    outcome = _collect(repo, _FakeClient(results=results))
    assert outcome.status == "failed"
    ledger = repo / "reports" / "credit" / "ledger.jsonl"
    assert json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])["status"] == "failed"


def test_collect_refuses_to_start_when_the_cost_cannot_be_priced(repo: Path) -> None:
    plan = _submitted_plan(repo)
    _write(
        repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML.replace("claude-sonnet-5-5", "claude-other")
    )
    client = _FakeClient(results=[_answer_all(r) for r in plan.requests])
    with pytest.raises(triage.TriageError, match="claude-sonnet-5-5"):
        _collect(repo, client)
    assert not (repo / "docs" / "analysis").exists()  # no document without a ledger row to match


# --------------------------------------------------------------------------- #
# collect: refusing to join results to the wrong findings
# --------------------------------------------------------------------------- #
def test_collect_refuses_an_unknown_batch_id(repo: Path) -> None:
    _submit(repo)
    clock = _Clock()
    with pytest.raises(triage.TriageError, match="msgbatch_other"):
        triage.collect(
            "msgbatch_other",
            repo / "out.md",
            repo_root=repo,
            client=_FakeClient(),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )


def test_collect_refuses_an_audit_json_that_changed_since_submit(repo: Path) -> None:
    _submit(repo)
    audit = repo / "reports" / "dead_code" / f"{_FIXED_DATE}.json"
    rows = _audit_rows()
    rows[0]["lineno"] += 1
    audit.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(triage.TriageError, match="sha256"):
        _collect(repo, _FakeClient())


def test_collect_refuses_when_the_plan_drifted_since_submit(repo: Path) -> None:
    _submit(repo)
    _write(
        repo,
        DEFAULT_CONFIG_RELPATH,
        _WORKFORCE_YAML + "    batch_triage:\n        max_findings_per_request: 1\n",
    )
    with pytest.raises(triage.TriageError, match="max_findings_per_request"):
        _collect(repo, _FakeClient())


def test_collect_can_be_pointed_at_a_moved_audit_json(repo: Path) -> None:
    plan = _submitted_plan(repo)
    audit = repo / "reports" / "dead_code" / f"{_FIXED_DATE}.json"
    moved = repo / "elsewhere.json"
    moved.write_bytes(audit.read_bytes())
    audit.unlink()
    client = _FakeClient(results=[_answer_all(r) for r in plan.requests])

    with pytest.raises(triage.TriageError, match="not found"):
        _collect(repo, client)

    outcome = _collect(repo, client, findings_path=moved)
    assert sum(outcome.verdict_counts.values()) == 6


# --------------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------------- #
def _meta(**overrides: Any) -> triage.ReportMeta:
    base: dict[str, Any] = {
        "date": _FIXED_DATE,
        "batch_id": "msgbatch_test01",
        "model_requested": "claude-sonnet-5-5",
        "model_resolved": "claude-sonnet-5-5",
        "resolved_via": "models.retrieve",
        "findings_path": f"reports/dead_code/{_FIXED_DATE}.json",
        "findings_sha256": "ab" * 32,
        "vulture_version": "2.16",
        "request_count": 3,
        "request_problems": 0,
        "usage": triage.Usage(input_tokens=4000, output_tokens=600),
        "cost_usd": Decimal("0.007"),
        "workstream": "C",
        "run_id": "wsc-msgbatch_test01",
    }
    base.update(overrides)
    return triage.ReportMeta(**base)


def _verdict(
    name: str,
    verdict: str,
    *,
    path: str = f"{_SRC}/alpha/util.py",
    lineno: int = 1,
    confidence: int = 60,
    reason: str = "because",
    origin: str = "model",
) -> triage.Verdict:
    finding = _finding(path, lineno, name, "variable", confidence)
    return triage.Verdict(finding, verdict, reason, origin)


def test_render_orders_delete_verify_keep() -> None:
    verdicts = [
        _verdict("kept_one", "keep", lineno=1),
        _verdict("check_me", "verify", lineno=2),
        _verdict("remove_low", "delete", lineno=3, confidence=60),
        _verdict("kept_two", "keep", lineno=4),
        _verdict("remove_high", "delete", lineno=5, confidence=100),
        _verdict("check_me_too", "verify", lineno=6),
    ]

    document = triage.render_markdown(verdicts, _meta())

    delete_at = document.index("## Delete")
    verify_at = document.index("## Verify")
    keep_at = document.index("## Keep")
    assert delete_at < verify_at < keep_at
    # delete is ranked by confidence, highest first
    assert document.index("remove_high") < document.index("remove_low")
    assert delete_at < document.index("remove_high") < verify_at
    assert verify_at < document.index("check_me") < keep_at
    assert document.index("kept_one") > keep_at
    assert document.index("kept_two") > keep_at


def test_render_breaks_confidence_ties_by_location_so_the_document_is_stable() -> None:
    verdicts = [
        _verdict("z_last", "delete", path=f"{_SRC}/b.py", lineno=1),
        _verdict("a_second", "delete", path=f"{_SRC}/a.py", lineno=9),
        _verdict("a_first", "delete", path=f"{_SRC}/a.py", lineno=2),
    ]
    first = triage.render_markdown(verdicts, _meta())
    second = triage.render_markdown(list(reversed(verdicts)), _meta())
    assert first == second
    assert first.index("a_first") < first.index("a_second") < first.index("z_last")


def test_render_header_records_the_audit_hash_the_vulture_version_and_the_provenance() -> None:
    document = triage.render_markdown([_verdict("x", "keep")], _meta())
    assert "ab" * 32 in document
    assert "2.16" in document
    assert f"reports/dead_code/{_FIXED_DATE}.json" in document
    assert "msgbatch_test01" in document
    assert "claude-sonnet-5-5" in document
    assert "wsc-msgbatch_test01" in document
    assert "$0.0070" in document
    assert "no change" in document.lower()  # triage only: nothing was deleted


def test_render_counts_and_package_summary_add_up() -> None:
    verdicts = [
        _verdict("a", "delete", path=f"{_SRC}/alpha/x.py"),
        _verdict("b", "keep", path=f"{_SRC}/alpha/x.py"),
        _verdict("c", "verify", path=f"{_SRC}/beta/y.py"),
        _verdict("d", "verify", path=f"{_SRC}/beta/y.py", origin="fallback"),
    ]
    document = triage.render_markdown(verdicts, _meta())
    assert re.search(r"\| `?src/mousedroid/alpha`? \| 1 \| 0 \| 1 \| 2 \|", document)
    assert re.search(r"\| `?src/mousedroid/beta`? \| 0 \| 2 \| 0 \| 2 \|", document)
    assert "1 of 4" in document  # one verdict has no model answer behind it


def test_render_keeps_a_table_cell_from_breaking_the_table() -> None:
    nasty = _verdict("x", "keep", reason="a | b\nc <script>alert(1)</script> [l](http://e)")
    document = triage.render_markdown([nasty], _meta())
    row = next(line for line in document.splitlines() if "`x`" in line)
    assert row.count("|") == row.count("\\|") + 7  # the cell's own pipe is escaped, not a column
    assert "<script>" not in document


def test_render_of_an_empty_audit_is_still_a_valid_document() -> None:
    document = triage.render_markdown([], _meta())
    assert "## Delete" in document
    assert "## Verify" in document
    assert "## Keep" in document


def test_collect_writes_the_document_where_asked_and_creates_the_directory(repo: Path) -> None:
    plan = _submitted_plan(repo)
    target = repo / "docs" / "analysis" / "deeper" / "triage.md"
    client = _FakeClient(results=[_answer_all(r) for r in plan.requests])
    outcome = _collect(repo, client, out=target)
    assert outcome.out_path.resolve() == target.resolve()
    assert target.is_file()
    assert b"\r" not in target.read_bytes()  # LF in a tracked file
    audit = repo / "reports" / "dead_code" / f"{_FIXED_DATE}.json"
    assert hashlib.sha256(audit.read_bytes()).hexdigest() in target.read_text(encoding="utf-8")


def test_collect_defaults_the_document_name_to_the_run_date(repo: Path) -> None:
    plan = _submitted_plan(repo)
    client = _FakeClient(results=[_answer_all(r) for r in plan.requests])
    clock = _Clock()
    outcome = triage.collect(
        "msgbatch_test01",
        None,
        repo_root=repo,
        client=client,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    expected = repo / "docs" / "analysis" / f"dead-code-triage-{_FIXED_DATE}.md"
    assert outcome.out_path.resolve() == expected.resolve()


def test_the_vulture_version_is_the_installed_one() -> None:
    try:
        expected = metadata.version("vulture")
    except metadata.PackageNotFoundError:
        expected = "unknown"
    assert triage.vulture_version() == expected


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def test_main_submit_dry_run_prints_the_plan_and_the_estimate(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audit = _write_findings(repo)
    argv = ["submit", "--findings", str(audit), "--cap-usd", "15", "--dry-run"]
    code = triage.main([*argv, "--repo-root", str(repo)])
    out = capsys.readouterr().out
    assert code == 0
    assert "3 requests" in out
    assert "6 findings" in out
    assert "estimate" in out.lower()
    assert "$" in out
    assert "dry run" in out.lower()


def test_main_requires_an_explicit_cap_for_submit(repo: Path) -> None:
    argv = ["submit", "--findings", str(_write_findings(repo)), "--repo-root", str(repo)]
    with pytest.raises(SystemExit) as excinfo:
        triage.main(argv)
    assert excinfo.value.code == 2


def test_main_has_no_way_to_pass_an_api_key() -> None:
    for argv in (
        ["submit", "--findings", "f.json", "--cap-usd", "1", "--api-key", "x"],
        ["collect", "msgbatch_1", "--api-key", "x"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            triage.main(argv)
        assert excinfo.value.code == 2


def test_main_maps_a_refused_spend_to_a_distinct_exit_code(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(triage, "_default_scanner", lambda *_a, **_k: _clean_scanner)
    audit = _write_findings(repo)
    argv = ["submit", "--findings", str(audit), "--cap-usd", "0.01", "--repo-root", str(repo)]
    assert triage.main(argv) == 3
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "--cap-usd" in err  # it is the cap that refused, not the scanner


def test_main_reports_a_missing_audit_json_cleanly(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["submit", "--findings", str(repo / "nope.json"), "--cap-usd", "15"]
    assert triage.main([*argv, "--repo-root", str(repo)]) == 1
    assert "nope.json" in capsys.readouterr().err


def test_main_collect_no_wait_exits_4_while_the_batch_is_running(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _submit(repo)
    monkeypatch.setattr(triage, "_default_client", lambda: _FakeClient(statuses=["in_progress"]))
    code = triage.main(["collect", "msgbatch_test01", "--no-wait", "--repo-root", str(repo)])
    assert code == 4
    assert "msgbatch_test01" in capsys.readouterr().err


def test_main_submit_then_collect_end_to_end(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audit = _write_findings(repo)
    plan = _plan(repo)
    results = [_answer_all(r, {"orphan_helper": "delete"}) for r in plan.requests]
    client = _FakeClient(results=results)
    monkeypatch.setattr(triage, "_default_client", lambda: client)
    monkeypatch.setattr(triage, "_default_scanner", lambda *_a, **_k: _clean_scanner)

    submit_argv = ["submit", "--findings", str(audit), "--cap-usd", "15", "--repo-root", str(repo)]
    assert triage.main(submit_argv) == 0
    submitted = capsys.readouterr().out
    assert "msgbatch_test01" in submitted
    assert "collect msgbatch_test01" in submitted

    collect_argv = ["collect", "msgbatch_test01", "--poll-interval", "0.001"]
    assert triage.main([*collect_argv, "--repo-root", str(repo)]) == 0
    assert "dead-code-triage-2026-10-19.md" in capsys.readouterr().out
    document = (repo / "docs" / "analysis" / "dead-code-triage-2026-10-19.md").read_text(
        encoding="utf-8"
    )
    assert "`orphan_helper`" in document
    assert document.index("## Delete") < document.index("`orphan_helper`")


def test_main_collect_twice_says_the_ledger_was_not_charged_again(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _submitted_plan(repo)
    client = _FakeClient(results=[_answer_all(r) for r in plan.requests])
    monkeypatch.setattr(triage, "_default_client", lambda: client)
    argv = ["collect", "msgbatch_test01", "--repo-root", str(repo)]

    assert triage.main(argv) == 0
    assert "ledger: workstream C" in capsys.readouterr().out
    assert triage.main(argv) == 0
    assert "already recorded" in capsys.readouterr().out
    ledger = repo / "reports" / "credit" / "ledger.jsonl"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


# --------------------------------------------------------------------------- #
# the corners: input hygiene, helpers, defaults
# --------------------------------------------------------------------------- #
def test_a_leading_dot_slash_is_dropped_from_a_recorded_path() -> None:
    assert _finding("./src/mousedroid/a/b.py", 1, "x").filename == f"{_SRC}/a/b.py"


def test_build_request_with_no_findings_is_an_error_not_an_empty_prompt(repo: Path) -> None:
    with pytest.raises(triage.TriageError, match="no findings"):
        triage.build_request(f"{_SRC}/alpha", [], repo)


def test_a_split_package_whose_name_leaves_no_room_for_the_suffix_is_an_error(repo: Path) -> None:
    long_name = "p" * 59  # "dead-" + 59 characters is 64: it fits only without a "-2"
    rows = [_row(f"{_SRC}/{long_name}/m.py", n, f"s{n}") for n in range(1, 22)]
    with pytest.raises(triage.TriageError, match="64"):
        _plan(repo, rows)


def test_hints_are_found_once_per_package_not_once_per_request(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(f"{_SRC}/alpha/util.py", n, f"value_{n}", "variable") for n in range(1, 46)]
    asked: list[str] = []
    real = triage.collect_hints

    def _counting(repo_arg: Path, pkg: str) -> triage.PackageHints:
        asked.append(pkg)
        return real(repo_arg, pkg)

    monkeypatch.setattr(triage, "collect_hints", _counting)
    requests = triage.build_requests(_plan(repo, rows), repo, _settings(repo), scanner=None)
    assert len(requests) == 3  # one package, three requests ...
    assert asked == [f"{_SRC}/alpha"]  # ... and the package's hints were read once


def test_the_default_scanner_runs_the_repository_scanner_on_the_prompt(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _fake_scan(
        content: str, cfg: WorkforceConfig, *, repo_root: Path, target: str
    ) -> ScanOutcome:
        seen.update(content=content, cfg=cfg, repo_root=repo_root, target=target)
        return ScanOutcome(clean=False, available=True, detail="1 leak")

    monkeypatch.setattr(triage.secret_scan, "scan_content", _fake_scan)
    workforce = _workforce(repo)

    outcome = triage._default_scanner(workforce, repo)("some prompt")

    assert outcome.clean is False
    assert seen["content"] == "some prompt"
    assert seen["cfg"] is workforce
    assert seen["repo_root"] == repo


def test_the_default_scanner_stands_down_when_secret_scan_is_disabled(repo: Path) -> None:
    _write(repo, DEFAULT_CONFIG_RELPATH, _WORKFORCE_YAML + "secret_scan:\n    enabled: false\n")
    outcome = triage._default_scanner(_workforce(repo), repo)("anything")
    assert outcome.available is False
    assert outcome.clean is True


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), -1.0])
def test_submit_rejects_a_cap_that_is_not_a_finite_non_negative_number(
    repo: Path, cap: float
) -> None:
    client = MagicMock()
    with pytest.raises(triage.TriageError, match="cap-usd"):
        triage.submit(
            _write_findings(repo), cap, repo_root=repo, client=client, scanner=_clean_scanner
        )
    assert client.mock_calls == []


def test_a_cap_that_is_not_even_a_number_is_named_as_such() -> None:
    not_a_number: Any = "abc"
    with pytest.raises(triage.TriageError, match="must be a number"):
        triage._cap_decimal(not_a_number)


def test_submit_with_nothing_to_send_says_so_and_calls_nothing(repo: Path) -> None:
    client = MagicMock()
    for rows in ([_row("config/docker.env", 1, "leaky")], []):
        audit = _write_findings(repo, rows)
        with pytest.raises(triage.TriageError, match="nothing to send"):
            triage.submit(audit, 15.0, repo_root=repo, client=client, scanner=_clean_scanner)
    assert client.mock_calls == []


def test_the_answer_validator_agrees_with_the_schema_it_reads() -> None:
    item = {"name": "n", "filename": "f.py", "lineno": 3, "reason": "r", "verdict": "keep"}
    assert triage._schema_errors(triage.RESULT_SCHEMA, {"findings": [item]}, "$") == []
    bad = {**item, "lineno": "3", "verdict": "maybe"}
    errors = triage._schema_errors(triage.RESULT_SCHEMA, {"findings": [bad]}, "$")
    assert any("$.findings[0].lineno" in e for e in errors)
    assert any("$.findings[0].verdict" in e for e in errors)


def test_the_answer_validator_refuses_a_schema_type_it_does_not_implement() -> None:
    with pytest.raises(ValueError, match="boolean"):
        triage._schema_errors({"type": "boolean"}, True, "$")


def test_parse_answer_drops_entries_that_do_not_say_what_they_are_about() -> None:
    first, second = _two_findings()
    vague: list[Any] = [
        "just a string",
        {"name": 1},
        {"name": "x", "filename": "f.py", "lineno": True},
        {"name": "x", "filename": 5, "lineno": 1},
    ]
    text = _answer([*vague, _entry(first), _entry(second)])
    verdicts = triage.parse_answer(text, [first, second])
    assert [v.origin for v in verdicts] == ["model", "model"]


def test_hints_see_plain_protocol_attributes_and_ignore_a_decorator_they_cannot_name(
    repo: Path,
) -> None:
    _write(
        repo,
        f"{_SRC}/beta/odd.py",
        "from typing import Protocol\n\nregistry = {}\n\n\n"
        "class P(Protocol):\n    flag = True\n\n    def go(self): ...\n\n\n"
        "class C:\n    @registry['x']\n    def m(self): ...\n",
    )
    hints = triage.collect_hints(repo, f"{_SRC}/beta")
    assert dict(hints.protocols) == {"P": ("flag", "go")}
    assert hints.validators == ()


def test_hints_do_not_follow_a_package_name_out_of_the_repository(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "x.py").write_text("from typing import Protocol\nclass P(Protocol): ...\n")
    nothing = triage.PackageHints((), ())
    assert triage.collect_hints(repo, f"../{outside.name}") == nothing
    assert triage.collect_hints(repo, str(outside)) == nothing


def test_utc_now_is_timezone_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # drop the autouse clock pin: this one tests the real function
    now = triage._utc_now()
    assert now.utcoffset() == timedelta(0)


def test_a_failed_write_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out" / "doc.md"
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")

    def _refuse(_src: object, _dst: object) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(triage.os, "replace", _refuse)
    with pytest.raises(OSError, match="rename failed"):
        triage._atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in target.parent.iterdir()] == ["doc.md"]


def test_a_file_outside_the_repo_is_shown_by_its_name_only(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "audit.json"
    assert triage._display_path(elsewhere, repo) == "audit.json"


def test_vulture_version_is_unknown_when_vulture_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_name: str) -> str:
        raise metadata.PackageNotFoundError("vulture")

    monkeypatch.setattr(triage.metadata, "version", _missing)
    assert triage.vulture_version() == "unknown"


def test_collect_skips_a_corrupt_record_and_finds_the_right_one(repo: Path) -> None:
    plan = _submitted_plan(repo)
    batches = repo / "reports" / "credit" / "batches"
    (batches / "2026-10-18.json").write_text("{corrupt", encoding="utf-8")
    (batches / "2026-10-17.json").write_text("[1, 2]", encoding="utf-8")
    outcome = _collect(repo, _FakeClient(results=[_answer_all(r) for r in plan.requests]))
    assert sum(outcome.verdict_counts.values()) == 6


def test_collect_with_no_batch_ever_submitted_says_there_is_no_record(repo: Path) -> None:
    with pytest.raises(triage.TriageError, match="no batch record"):
        _collect(repo, _FakeClient())


def test_a_duplicate_result_for_one_custom_id_keeps_the_first(repo: Path) -> None:
    plan = _submitted_plan(repo)
    alpha = plan.requests[0]
    first = _answer_all(alpha, {"orphan_helper": "delete"})
    second = _answer_all(alpha, {"orphan_helper": "keep"})
    rest = [_answer_all(r) for r in plan.requests[1:]]

    outcome = _collect(repo, _FakeClient(results=[first, second, *rest]))

    assert outcome.verdict_counts["delete"] == 1


def test_the_answer_validator_leaves_extra_keys_alone_when_the_schema_does_not_forbid_them() -> (
    None
):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert triage._schema_errors(schema, {"a": "x", "extra": 1}, "$") == []
    closed = {**schema, "additionalProperties": False}
    assert triage._schema_errors(closed, {"a": "x", "extra": 1}, "$") == [
        '$: unexpected property "extra"'
    ]


def test_the_ledger_scan_for_a_run_tolerates_blank_lines(repo: Path) -> None:
    plan = _submitted_plan(repo)
    results = [_answer_all(r) for r in plan.requests]
    _collect(repo, _FakeClient(results=results))
    ledger = repo / "reports" / "credit" / "ledger.jsonl"
    ledger.write_text("\n\n" + ledger.read_text(encoding="utf-8"), encoding="utf-8")
    record_path = repo / "reports" / "credit" / "batches" / f"{_FIXED_DATE}.json"
    unmarked = _record(repo)
    unmarked["collected"] = None
    record_path.write_text(json.dumps(unmarked), encoding="utf-8")

    again = _collect(repo, _FakeClient(results=results))

    assert again.ledger_row is None
    assert len([ln for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]) == 1


def test_collect_adds_its_row_to_a_ledger_that_already_holds_other_runs(repo: Path) -> None:
    plan = _submitted_plan(repo)
    ledger = repo / "reports" / "credit" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    earlier = {"ws": "C", "run_id": "wsc-earlier", "cost_usd": 0.5}
    ledger.write_text(json.dumps(earlier) + "\n", encoding="utf-8")

    outcome = _collect(repo, _FakeClient(results=[_answer_all(r) for r in plan.requests]))

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [r["run_id"] for r in rows] == ["wsc-earlier", "wsc-msgbatch_test01"]
    assert outcome.ledger_row is not None
    assert outcome.ledger_row["cumulative_ws_usd"] > 0.5


def test_default_client_explains_a_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail
    with pytest.raises(triage.TriageError, match="anthropic SDK is not installed"):
        triage._default_client()


def test_resolve_model_without_a_models_api_uses_the_configured_id() -> None:
    client = types.SimpleNamespace(messages=types.SimpleNamespace())
    assert triage._resolve_model(client, "m") == ("m", "config")


def test_resolve_model_warns_about_a_deprecated_model_and_goes_on(
    capsys: pytest.CaptureFixture[str],
) -> None:
    info = types.SimpleNamespace(id="m-2", lifecycle="deprecated")
    client = types.SimpleNamespace(models=types.SimpleNamespace(retrieve=lambda _id: info))
    assert triage._resolve_model(client, "m") == ("m-2", "models.retrieve")
    assert "deprecated" in capsys.readouterr().err


def test_resolve_model_ignores_a_reply_that_carries_no_id() -> None:
    info = types.SimpleNamespace(lifecycle="active")
    client = types.SimpleNamespace(models=types.SimpleNamespace(retrieve=lambda _id: info))
    assert triage._resolve_model(client, "m") == ("m", "config")


def test_a_succeeded_result_with_no_text_block_leaves_its_findings_to_verify(repo: Path) -> None:
    plan = _submitted_plan(repo)
    thinking_only = types.SimpleNamespace(type="thinking", thinking="")
    empty_text = types.SimpleNamespace(type="text", text=None)
    message = types.SimpleNamespace(
        content=[thinking_only, empty_text], stop_reason="end_turn", usage=_usage(10, 5)
    )
    odd = types.SimpleNamespace(
        custom_id="dead-alpha", result=types.SimpleNamespace(type="succeeded", message=message)
    )
    others = [_answer_all(r) for r in plan.requests if r.custom_id != "dead-alpha"]

    outcome = _collect(repo, _FakeClient(results=[odd, *others]))

    assert outcome.verdict_counts["verify"] == 4
    assert "no text block" in outcome.out_path.read_text(encoding="utf-8")
    assert outcome.usage.input_tokens >= 10  # the tokens were still billed


def test_cost_is_rounded_up_to_the_micro_dollar_never_down() -> None:
    price = ModelPrice(input_usd_per_mtok=0.3, output_usd_per_mtok=0.3)
    cost = triage.usage_cost_usd(triage.Usage(input_tokens=1), price, 1.0)
    assert cost == Decimal("0.000001")  # 3e-7 exactly, rounded up from 0.0000003


# --------------------------------------------------------------------------- #
# the REAL anthropic SDK (skipped without the optional extra)
# --------------------------------------------------------------------------- #
def _real_sdk() -> Any:
    return pytest.importorskip("anthropic", reason="optional extra: pip install -e '.[anthropic]'")


def _real_http() -> Any:
    try:  # anthropic>=1.0 is built on httpx2; older releases on httpx
        import httpx2 as httpx_lib
    except ImportError:  # pragma: no cover - only on anthropic<1.0
        httpx_lib = pytest.importorskip("httpx")
    return httpx_lib


_BATCH_JSON: dict[str, Any] = {
    "id": "msgbatch_wire",
    "type": "message_batch",
    "processing_status": "in_progress",
    "request_counts": {
        "processing": 3,
        "succeeded": 0,
        "errored": 0,
        "canceled": 0,
        "expired": 0,
    },
    "created_at": "2026-10-19T14:02:11Z",
    "expires_at": "2026-10-20T14:02:11Z",
    "ended_at": None,
    "archived_at": None,
    "cancel_initiated_at": None,
    "results_url": None,
}


def _real_client(handler: Callable[[Any], Any]) -> Any:
    anthropic = _real_sdk()
    http = _real_http()
    return anthropic.Anthropic(
        api_key="test-key",
        base_url="http://mock.invalid",
        http_client=anthropic.DefaultHttpxClient(transport=http.MockTransport(handler)),
        max_retries=0,
    )


def test_default_client_is_the_sdks_client_built_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic = _real_sdk()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # never a real one: nothing is sent
    client = triage._default_client()
    assert isinstance(client, anthropic.Anthropic)
    assert callable(client.messages.batches.create)
    assert callable(client.messages.batches.results)


def test_requests_use_only_parameters_the_installed_sdk_declares(repo: Path) -> None:
    _real_sdk()
    from anthropic.types import message_create_params
    from anthropic.types.messages import batch_create_params
    from anthropic.types.output_config_param import OutputConfigParam

    declared = set(message_create_params.MessageCreateParamsNonStreaming.__annotations__)
    for request in _requests(repo):
        assert set(request) == set(batch_create_params.Request.__annotations__)
        params = request["params"]
        assert {"max_tokens", "messages", "model"} <= set(params)
        assert set(params) <= declared, set(params) - declared
        assert set(params["output_config"]) <= set(OutputConfigParam.__annotations__)


def _literal_values(annotation: Any) -> set[Any]:
    """Every value of every ``Literal`` inside ``annotation`` (through ``Optional`` / ``Union``)."""
    if typing.get_origin(annotation) is typing.Literal:
        return set(typing.get_args(annotation))
    found: set[Any] = set()
    for arg in typing.get_args(annotation):
        found |= _literal_values(arg)
    return found


def test_the_effort_and_thinking_modes_the_config_allows_exist_in_the_sdk(repo: Path) -> None:
    _real_sdk()
    from anthropic.types.output_config_param import OutputConfigParam
    from anthropic.types.thinking_config_adaptive_param import ThinkingConfigAdaptiveParam
    from anthropic.types.thinking_config_between_tools_param import (
        ThinkingConfigBetweenToolsParam,
    )

    sdk_efforts = _literal_values(typing.get_type_hints(OutputConfigParam)["effort"])
    config_efforts = _literal_values(BatchTriageConfig.model_fields["effort"].annotation)
    assert config_efforts <= sdk_efforts

    sdk_modes = {
        _literal_values(typing.get_type_hints(cls)["type"]).pop()
        for cls in (ThinkingConfigAdaptiveParam, ThinkingConfigBetweenToolsParam)
    }
    config_modes = _literal_values(BatchTriageConfig.model_fields["thinking"].annotation)
    assert config_modes == sdk_modes

    finding = _finding(f"{_SRC}/beta/engine.py", 20, "value_20", "variable")
    for mode in sorted(config_modes):
        settings = BatchTriageConfig(thinking=mode)
        request = triage.build_request(f"{_SRC}/beta", [finding], repo, settings=settings)
        assert request["params"]["thinking"] == {"type": mode}


def test_the_sdk_puts_the_structured_output_request_on_the_wire(repo: Path) -> None:
    http = _real_http()
    bodies: list[dict[str, Any]] = []

    def _handler(request: Any) -> Any:
        bodies.append(json.loads(request.content))
        return http.Response(200, json=_BATCH_JSON)

    client = _real_client(_handler)
    requests = _requests(repo)
    batch = client.messages.batches.create(requests=requests)

    assert batch.id == "msgbatch_wire"
    sent = bodies[-1]["requests"]
    assert [r["custom_id"] for r in sent] == [r["custom_id"] for r in requests]
    params = sent[0]["params"]
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["output_config"]["format"]["schema"] == triage.RESULT_SCHEMA
    assert params["output_config"]["effort"] == "low"
    assert "temperature" not in params


def test_submit_works_through_the_real_sdk_client_shape(repo: Path) -> None:
    """``submit`` reads ``.id`` off the SDK's ``MessageBatch`` and ``.id`` off ``ModelInfo``."""
    http = _real_http()
    paths: list[str] = []
    model_json = {
        "id": "claude-sonnet-5-5",
        "type": "model",
        "display_name": "Claude Sonnet 5.5",
        "created_at": "2026-09-01T00:00:00Z",
        "lifecycle": "active",
    }

    def _handler(request: Any) -> Any:
        paths.append(request.url.path)
        if request.url.path.startswith("/v1/models/"):
            return http.Response(200, json=model_json)
        return http.Response(200, json={**_BATCH_JSON, "id": "msgbatch_wire2"})

    outcome = _submit(repo, _real_client(_handler))

    assert outcome.batch_id == "msgbatch_wire2"
    assert outcome.resolved_via == "models.retrieve"
    assert paths == ["/v1/models/claude-sonnet-5-5", "/v1/messages/batches"]


def test_a_failing_create_is_sent_once_through_the_real_sdk(repo: Path) -> None:
    """The SDK retries a 5xx by default; for a call that spends money that would be a 2nd batch."""
    anthropic = _real_sdk()
    http = _real_http()
    paths: list[str] = []

    def _handler(request: Any) -> Any:
        paths.append(request.url.path)
        error = {"type": "api_error", "message": "boom"}
        return http.Response(500, json={"type": "error", "error": error})

    client = anthropic.Anthropic(  # default max_retries: only `submit` switches retries off
        api_key="test-key",
        base_url="http://mock.invalid",
        http_client=anthropic.DefaultHttpxClient(transport=http.MockTransport(_handler)),
    )

    with pytest.raises(triage.TriageError, match="may exist"):
        _submit(repo, client)

    assert paths.count("/v1/messages/batches") == 1


def test_submit_then_collect_through_the_real_sdk_over_a_mock_http_server(repo: Path) -> None:
    """Every call the tool makes, as real SDK calls whose HTTP layer is a fixed script.

    This is the only test where ``retrieve`` / ``results`` return the SDK's own objects
    decoded from bytes (the JSONL stream included), so it is the one that proves the
    attribute names the tool reads exist on them.
    """
    http = _real_http()
    plan = _plan(repo)
    batch_id = "msgbatch_wire3"
    results_url = f"http://mock.invalid/v1/messages/batches/{batch_id}/results"
    jsonl: list[str] = []
    for planned in plan.requests:
        entries = [
            _entry(f, "delete" if f.name == "orphan_helper" else "keep", f"why {f.name}")
            for f in planned.findings
        ]
        message = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": _answer(entries)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 100},
        }
        result = {"type": "succeeded", "message": message}
        jsonl.append(json.dumps({"custom_id": planned.custom_id, "result": result}))
    jsonl.reverse()  # the API promises no order
    model_json = {
        "id": "claude-sonnet-5-5",
        "type": "model",
        "display_name": "Claude Sonnet 5.5",
        "created_at": "2026-09-01T00:00:00Z",
        "lifecycle": "active",
    }
    counts = {"processing": 0, "succeeded": 3, "errored": 0, "canceled": 0, "expired": 0}
    ended = {
        **_BATCH_JSON,
        "id": batch_id,
        "processing_status": "ended",
        "ended_at": "2026-10-19T15:00:00Z",
        "results_url": results_url,
        "request_counts": counts,
    }

    calls: list[str] = []

    def _handler(request: Any) -> Any:
        path = request.url.path
        calls.append(f"{request.method} {path}")
        if path.startswith("/v1/models/"):
            return http.Response(200, json=model_json)
        if request.method == "POST" and path == "/v1/messages/batches":
            return http.Response(200, json={**_BATCH_JSON, "id": batch_id})
        if path == f"/v1/messages/batches/{batch_id}":
            return http.Response(200, json=ended)
        if path == f"/v1/messages/batches/{batch_id}/results":
            body = ("\n".join(jsonl) + "\n").encode("utf-8")
            return http.Response(200, content=body, headers={"content-type": "application/binary"})
        return http.Response(404, json={"type": "error", "error": {"type": "not_found_error"}})

    client = _real_client(_handler)

    submitted = _submit(repo, client)
    outcome = _collect(repo, client, batch_id=batch_id)

    assert f"GET /v1/messages/batches/{batch_id}/results" in calls  # the JSONL really was streamed
    assert calls.count("POST /v1/messages/batches") == 1
    assert submitted.batch_id == batch_id
    assert outcome.verdict_counts == {"delete": 1, "verify": 0, "keep": 5}
    assert outcome.usage.input_tokens == 3000
    assert outcome.usage.output_tokens == 300
    assert outcome.status == "ok"
    document = outcome.out_path.read_text(encoding="utf-8")
    assert re.search(r"`orphan_helper`.*service\.py", document)
    assert "why value_30" in document


def test_results_in_the_sdks_own_shape_are_read_correctly(repo: Path) -> None:
    """``MessageBatchIndividualResponse`` objects built by the SDK, not by a fake of them."""
    _real_sdk()
    from anthropic.types.messages.message_batch_individual_response import (
        MessageBatchIndividualResponse,
    )

    plan = _submitted_plan(repo)

    def _line(planned: triage.PlannedRequest) -> Any:
        entries = [_entry(f, "keep", "alive") for f in planned.findings]
        return MessageBatchIndividualResponse.model_validate(
            {
                "custom_id": planned.custom_id,
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-5-5",
                        "content": [{"type": "text", "text": _answer(entries)}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                    },
                },
            }
        )

    errored = MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": plan.requests[1].custom_id,
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                    "request_id": "req_1",
                },
            },
        }
    )
    results = [_line(plan.requests[0]), errored, _line(plan.requests[2])]
    outcome = _collect(repo, _FakeClient(results=results))

    answered = len(plan.requests[0].findings) + len(plan.requests[2].findings)
    assert outcome.verdict_counts["keep"] == answered
    assert outcome.verdict_counts["verify"] == len(plan.requests[1].findings)
    assert outcome.usage.input_tokens == 200
    assert outcome.usage.output_tokens == 40
    assert "overloaded_error" in outcome.out_path.read_text(encoding="utf-8")
