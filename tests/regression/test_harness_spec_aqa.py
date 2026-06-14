# tests/regression/test_harness_spec_aqa.py
"""AQA: spec-driven harness (HARNESS_SPEC.md / features.yaml) hygiene.

Locks the contracts a careless edit could break, so the harness folds into the
project's own regression gate (scripts/ci.sh) rather than relying solely on the
standalone .github/workflows/harness.yml job:

  * features.yaml parses and validates against features.schema.json
  * the dependency DAG is acyclic with no dangling depends_on edges
  * every referenced scripts/validations/*.sh exists and is non-empty
  * every `done` feature carries a validation_command + implemented_in (schema
    half of the Golden Rule)
  * the runner modules (validate.py, select_next.py) import as files

See ADR-012 for the rationale.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEATURES = _REPO_ROOT / "features.yaml"
_SCHEMA = _REPO_ROOT / "features.schema.json"
_SPEC = _REPO_ROOT / "HARNESS_SPEC.md"


def _load_features() -> list[dict]:
    return yaml.safe_load(_FEATURES.read_text())["features"]


def test_core_harness_files_exist() -> None:
    for path in (
        _FEATURES,
        _SCHEMA,
        _SPEC,
        _REPO_ROOT / "scripts" / "validate.py",
        _REPO_ROOT / "scripts" / "select_next.py",
    ):
        assert path.is_file(), f"missing core harness file: {path}"


def test_features_validate_against_schema() -> None:
    jsonschema = __import__("jsonschema")
    schema = json.loads(_SCHEMA.read_text())
    feats = _load_features()
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors({"features": feats}),
        key=lambda e: list(e.path),
    )
    assert errors == [], "schema errors:\n" + "\n".join(
        f"  {list(e.path)}: {e.message}" for e in errors
    )


def test_feature_ids_are_unique() -> None:
    ids = [f["id"] for f in _load_features()]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate feature ids: {sorted(dupes)}"


def test_dag_has_no_dangling_edges_or_cycles() -> None:
    feats = _load_features()
    by_id = {f["id"]: f for f in feats}

    dangling = [(f["id"], d) for f in feats for d in f.get("depends_on", []) if d not in by_id]
    assert not dangling, f"dangling depends_on edges: {dangling}"

    # DFS three-colour cycle detection (mirrors scripts/validate.py).
    white, grey, black = 0, 1, 2
    color = {f["id"]: white for f in feats}
    cycles: list[str] = []

    def visit(node: str, stack: list[str]) -> None:
        color[node] = grey
        for dep in by_id[node].get("depends_on", []):
            if dep not in by_id:
                continue
            if color[dep] == grey:
                cycles.append(" -> ".join([*stack, dep]))
            elif color[dep] == white:
                visit(dep, [*stack, dep])
        color[node] = black

    for f in feats:
        if color[f["id"]] == white:
            visit(f["id"], [f["id"]])
    assert not cycles, f"dependency cycles: {cycles}"


def test_done_features_have_command_and_provenance() -> None:
    for f in _load_features():
        if f["status"] != "done":
            continue
        assert f.get("validation_command"), f"{f['id']}: done without validation_command"
        assert f.get("implemented_in"), f"{f['id']}: done without implemented_in"


def test_referenced_validation_scripts_exist() -> None:
    for f in _load_features():
        cmd = f.get("validation_command") or ""
        for token in cmd.split():
            if token.startswith("scripts/validations/") and token.endswith(".sh"):
                script = _REPO_ROOT / token
                assert script.is_file(), f"{f['id']}: missing validation script {token}"
                assert script.stat().st_size > 0, f"{f['id']}: empty validation script {token}"


def test_runner_modules_import_as_files() -> None:
    for name in ("validate", "select_next"):
        path = _REPO_ROOT / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_harness_{name}", path)
        assert spec is not None, f"cannot load {path}"
        assert spec.loader is not None, f"no loader for {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main"), f"{name}.py has no main()"
