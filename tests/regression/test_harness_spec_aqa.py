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
import re
import shlex
from pathlib import Path

import jsonschema
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEATURES = _REPO_ROOT / "features.yaml"
_SCHEMA = _REPO_ROOT / "features.schema.json"
_SPEC = _REPO_ROOT / "HARNESS_SPEC.md"


# Full 40-char git object name. Abbreviated SHAs are rejected on purpose:
# they can grow ambiguous as history accumulates.
_HEX_SHA = re.compile(r"[0-9a-f]{40}")


def _load_features() -> list[dict]:
    return yaml.safe_load(_FEATURES.read_text(encoding="utf-8"))["features"]


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
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
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
    # Single canonical DAG implementation lives in mousedroid.harness.spec
    # (no duplicated DFS to drift from scripts/validate.py).
    from mousedroid.harness.spec import check_dag

    errs = check_dag(_load_features())
    assert errs == [], "DAG integrity errors:\n" + "\n".join(f"  {e}" for e in errs)


def test_done_features_have_command_and_provenance() -> None:
    for f in _load_features():
        if f["status"] != "done":
            continue
        assert f.get("validation_command"), f"{f['id']}: done without validation_command"
        assert f.get("implemented_in"), f"{f['id']}: done without implemented_in"


def test_done_features_pin_a_hex_sha_not_a_branch_name() -> None:
    """`implemented_in` on a done feature is SHAPED like a full commit SHA.

    HARNESS_SPEC.md and .claude/skills/openspec-change/SKILL.md both state the
    rule ("a hex commit SHA, never a branch name"), and feature-closeout calls a
    branch-name pin debt "wearing a different hat" -- but nothing enforced it
    catalog-wide. A branch name satisfies the truthiness check above, then goes
    unresolvable the moment the branch is deleted, silently breaking the
    nightly `validate.py --tier fast,slow --strict-git` run.

    Scope, stated plainly because it is easy to over-read: this is a FORMAT
    check and nothing more. It does not verify that the SHA resolves, that it
    is an ancestor of any branch, or that it contains the work the feature
    claims. `175606b052bac2b613144ffca10c6a4982a7def8` -- a resolvable,
    well-formed SHA naming the *branch point*, which contained neither
    change -- shipped past this very assertion earlier in F-028/F-029's own
    history. Read it as "not a branch name", never as provenance.

    Resolvability is deliberately left to `validate.py --strict-git` in the
    nightly, which runs against a full clone. Asserting it here would be flaky
    locally and vacuous in CI: this repo is routinely shallow-cloned, so
    `git cat-file` on an older SHA fails for want of history rather than for
    want of a correct pin.
    """
    offenders = [
        f"{f['id']}: implemented_in={f.get('implemented_in')!r}"
        for f in _load_features()
        if f["status"] == "done" and not _HEX_SHA.fullmatch(str(f.get("implemented_in") or ""))
    ]
    assert offenders == [], (
        "done features must pin a full 40-character hex commit SHA:\n"
        + "\n".join(f"  {o}" for o in offenders)
    )


def test_referenced_validation_scripts_exist() -> None:
    for f in _load_features():
        cmd = f.get("validation_command") or ""
        # validation_command is a shell string, so tokenize the way a shell
        # would (shlex strips quotes); normalise a leading ./ so spellings like
        # `bash "./scripts/validations/F-001.sh"` are still existence-checked
        # instead of silently skipped by a naive str.split().
        for token in shlex.split(cmd):
            normalised = token.removeprefix("./")
            if normalised.startswith("scripts/validations/") and normalised.endswith(".sh"):
                script = _REPO_ROOT / normalised
                assert script.is_file(), f"{f['id']}: missing validation script {normalised}"
                assert script.stat().st_size > 0, f"{f['id']}: empty validation script {normalised}"


def test_runner_modules_import_as_files() -> None:
    for name in ("validate", "select_next"):
        path = _REPO_ROOT / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_harness_{name}", path)
        assert spec is not None, f"cannot load {path}"
        assert spec.loader is not None, f"no loader for {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "main"), f"{name}.py has no main()"
