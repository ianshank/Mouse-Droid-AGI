"""Unit tests for the spec-driven harness core (:mod:`mousedroid.harness.spec`).

These cover the enforcement logic that previously lived untested in
``scripts/validate.py`` / ``scripts/select_next.py`` (ADR-012): catalog loading,
schema validation, DAG integrity (dangling edges + cycles), git provenance,
command execution, tier-gated orchestration, and DAG-aware selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from mousedroid.harness import spec

Feature = dict[str, Any]


def _feat(fid: str, **kw: Any) -> Feature:
    """Build a minimal valid feature, overridable via kwargs."""
    base: Feature = {
        "id": fid,
        "name": f"feature {fid}",
        "category": "functional",
        "priority": "high",
        "status": "todo",
        "verification": ["does a thing"],
        "depends_on": [],
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# load_features
# --------------------------------------------------------------------------- #
def test_load_features_reads_list(tmp_path: Path) -> None:
    p = tmp_path / "features.yaml"
    p.write_text("features:\n  - id: F-001\n    name: x\n")
    feats = spec.load_features(p)
    assert feats == [{"id": "F-001", "name": "x"}]


@pytest.mark.parametrize("body", ["", "[]", "just_a_string\n", "features: 5\n"])
def test_load_features_rejects_malformed(tmp_path: Path, body: str) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(body)
    with pytest.raises(ValueError, match="features"):
        spec.load_features(p)


# --------------------------------------------------------------------------- #
# check_schema
# --------------------------------------------------------------------------- #
def test_check_schema_accepts_valid() -> None:
    schema_path = Path(__file__).resolve().parents[3] / "features.schema.json"
    feats = [_feat("F-001", status="todo")]
    assert spec.check_schema(feats, schema_path) == []


def test_check_schema_reports_violations(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "features": {
                        "type": "array",
                        "items": {"type": "object", "required": ["id"]},
                    }
                },
            }
        )
    )
    errs = spec.check_schema([{"name": "no id"}], schema_path)
    assert errs
    assert "schema:" in errs[0]


def test_check_schema_missing_jsonschema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    with pytest.raises(ModuleNotFoundError):
        spec.check_schema([], "ignored.json")


# --------------------------------------------------------------------------- #
# check_dag
# --------------------------------------------------------------------------- #
def test_check_dag_clean() -> None:
    feats = [_feat("F-001"), _feat("F-002", depends_on=["F-001"])]
    assert spec.check_dag(feats) == []


def test_check_dag_dangling_edge() -> None:
    errs = spec.check_dag([_feat("F-002", depends_on=["F-099"])])
    assert any("unknown id F-099" in e for e in errs)


def test_check_dag_self_cycle() -> None:
    errs = spec.check_dag([_feat("F-001", depends_on=["F-001"])])
    assert any("cycle" in e for e in errs)


def test_check_dag_multi_node_cycle() -> None:
    feats = [
        _feat("F-001", depends_on=["F-002"]),
        _feat("F-002", depends_on=["F-003"]),
        _feat("F-003", depends_on=["F-001"]),
    ]
    assert any("cycle" in e for e in spec.check_dag(feats))


def test_check_dag_reports_non_mapping_item() -> None:
    # Malformed catalog (schema validation skipped/unavailable): a non-mapping
    # entry must be reported, not raise.
    malformed: list[Any] = ["not-a-feature", _feat("F-001")]
    errs = spec.check_dag(malformed)
    assert any("feature[0] is not an object" in e for e in errs)


def test_check_dag_reports_missing_id() -> None:
    errs = spec.check_dag([{"name": "no id"}, {"id": ""}])
    assert sum("missing valid id" in e for e in errs) == 2


def test_check_dag_reports_duplicate_id() -> None:
    errs = spec.check_dag([_feat("F-001"), _feat("F-001")])
    assert any("duplicate feature id F-001" in e for e in errs)


# --------------------------------------------------------------------------- #
# git_rev_ok
# --------------------------------------------------------------------------- #
def test_git_rev_ok_resolves_head() -> None:
    assert spec.git_rev_ok("HEAD", cwd=Path(__file__).resolve().parents[3]) is True


@pytest.mark.parametrize("ref", [None, "", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"])
def test_git_rev_ok_rejects_bad_refs(ref: str | None) -> None:
    assert spec.git_rev_ok(ref, cwd=Path(__file__).resolve().parents[3]) is False


# --------------------------------------------------------------------------- #
# run_validation
# --------------------------------------------------------------------------- #
def test_run_validation_success() -> None:
    assert spec.run_validation(_feat("F-001", validation_command="true")) is None


def test_run_validation_failure_includes_tail() -> None:
    # Drive the failure through the current interpreter rather than a bash-only
    # one-liner so the test is shell-portable (validation_command runs under
    # cmd.exe on Windows, where `echo ... >&2; exit 3` neither fails nor writes
    # to stderr). stderr is merged into stdout, so the marker lands in the tail.
    cmd = f'"{sys.executable}" -c "import sys; sys.stderr.write(\'boom\'); sys.exit(3)"'
    err = spec.run_validation(_feat("F-001", validation_command=cmd))
    assert err is not None
    assert "F-001" in err
    assert "(3)" in err
    assert "boom" in err


def test_run_validation_missing_command() -> None:
    err = spec.run_validation(_feat("F-001"))
    assert err is not None
    assert "no validation_command" in err


def test_run_validation_timeout() -> None:
    # A command that sleeps longer than the (tiny) timeout is killed and reported,
    # not left to hang the harness. Driven through the interpreter for portability.
    cmd = f'"{sys.executable}" -c "import time; time.sleep(5)"'
    err = spec.run_validation(_feat("F-001", validation_command=cmd), timeout=0.5)
    assert err is not None
    assert "timed out" in err


# --------------------------------------------------------------------------- #
# run_features (orchestration, injected runner/rev_checker — no subprocess)
# --------------------------------------------------------------------------- #
def _ok_runner(_f: Feature) -> str | None:
    return None


def _ok_rev(_r: str | None) -> bool:
    return True


def test_run_features_runs_only_selected_tier() -> None:
    feats = [
        _feat("F-001", status="done", tier="fast", implemented_in="x"),
        _feat("F-002", status="done", tier="slow", implemented_in="x"),
        _feat("F-003", status="todo", tier="fast"),
    ]
    res = spec.run_features(feats, None, {"fast"}, runner=_ok_runner, rev_checker=_ok_rev)
    assert res.ok
    assert res.ran == 1
    assert res.skipped == 1
    assert res.done == 2


def test_run_features_default_tier_when_omitted() -> None:
    feats = [_feat("F-001", status="done", implemented_in="x")]  # no tier -> fast
    res = spec.run_features(feats, None, {"fast"}, runner=_ok_runner, rev_checker=_ok_rev)
    assert res.ran == 1


def test_run_features_collects_command_failures() -> None:
    feats = [_feat("F-001", status="done", tier="fast", implemented_in="x")]
    res = spec.run_features(
        feats, None, {"fast"}, runner=lambda f: f"{f['id']}: boom", rev_checker=_ok_rev
    )
    assert not res.ok
    assert res.errors == ["F-001: boom"]


def test_run_features_strict_git_promotes_warning_to_error() -> None:
    feats = [_feat("F-001", status="done", tier="fast", implemented_in="nope")]
    bad_rev = lambda _r: False  # noqa: E731

    lenient = spec.run_features(feats, None, {"fast"}, runner=_ok_runner, rev_checker=bad_rev)
    assert lenient.ok
    assert any("resolvable git ref" in w for w in lenient.warnings)

    strict = spec.run_features(
        feats, None, {"fast"}, strict_git=True, runner=_ok_runner, rev_checker=bad_rev
    )
    assert not strict.ok
    assert any("resolvable git ref" in e for e in strict.errors)


def test_run_features_schema_missing_lib_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # The harness is an enforced gate: a requested schema check with jsonschema
    # unavailable is a hard error, never a silently-skipped false-green run.
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    res = spec.run_features(
        [_feat("F-001")], "schema.json", {"fast"}, runner=_ok_runner, rev_checker=_ok_rev
    )
    assert not res.ok
    assert any("jsonschema not installed" in e for e in res.errors)


def test_run_features_short_circuits_on_dag_errors() -> None:
    # A structurally invalid catalog must not drive validation_command execution.
    feats = [_feat("F-001", status="done", tier="fast", depends_on=["F-404"])]

    def _boom_runner(_f: Feature) -> str | None:  # pragma: no cover - must not run
        raise AssertionError("runner must not execute when the DAG is invalid")

    res = spec.run_features(feats, None, {"fast"}, runner=_boom_runner, rev_checker=_ok_rev)
    assert not res.ok
    assert res.ran == 0
    assert res.skipped == 1
    assert any("unknown id F-404" in e for e in res.errors)


# --------------------------------------------------------------------------- #
# select_next
# --------------------------------------------------------------------------- #
def test_select_next_resumes_in_progress_by_priority() -> None:
    feats = [
        _feat("F-001", status="in_progress", priority="high"),
        _feat("F-002", status="in_progress", priority="critical"),
    ]
    sel = spec.select_next(feats)
    assert sel.kind == "resume"
    assert sel.feature is not None
    assert sel.feature["id"] == "F-002"


def test_select_next_picks_ready_highest_priority() -> None:
    feats = [
        _feat("F-001", status="done", implemented_in="x"),
        _feat("F-002", status="todo", priority="medium", depends_on=["F-001"]),
        _feat("F-003", status="todo", priority="critical", depends_on=["F-001"]),
    ]
    sel = spec.select_next(feats)
    assert sel.kind == "ready"
    assert sel.feature is not None
    assert sel.feature["id"] == "F-003"


def test_select_next_reports_blocked_with_unmet_deps() -> None:
    feats = [
        _feat("F-001", status="todo"),  # not done -> blocks F-002
        _feat("F-002", status="todo", depends_on=["F-001"]),
    ]
    # Only F-002 is passed in, so its F-001 dependency is absent (not done):
    # select_next must report F-002 as blocked on its unmet dependency.
    sel = spec.select_next([feats[1]])
    assert sel.kind == "blocked"
    assert sel.blocked == [("F-002", ["F-001"])]


def test_select_next_complete_when_no_todo() -> None:
    sel = spec.select_next([_feat("F-001", status="done", implemented_in="x")])
    assert sel.kind == "complete"


def test_select_next_tolerates_malformed() -> None:
    # A malformed entry (no id) must not raise; the well-formed todo is selected.
    sel = spec.select_next([{"status": "todo"}, _feat("F-001", status="todo")])
    assert sel.kind == "ready"
