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
    err = spec.run_validation(_feat("F-001", validation_command="echo boom >&2; exit 3"))
    assert err is not None
    assert "F-001" in err
    assert "(3)" in err
    assert "boom" in err


def test_run_validation_missing_command() -> None:
    err = spec.run_validation(_feat("F-001"))
    assert err is not None
    assert "no validation_command" in err


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


def test_run_features_schema_missing_lib_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    res = spec.run_features(
        [_feat("F-001")], "schema.json", {"fast"}, runner=_ok_runner, rev_checker=_ok_rev
    )
    assert any("jsonschema not installed" in w for w in res.warnings)


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
    # F-001 itself is ready (no deps), so it is selected first.
    sel = spec.select_next([feats[1]])
    assert sel.kind == "blocked"
    assert sel.blocked == [("F-002", ["F-001"])]


def test_select_next_complete_when_no_todo() -> None:
    sel = spec.select_next([_feat("F-001", status="done", implemented_in="x")])
    assert sel.kind == "complete"
