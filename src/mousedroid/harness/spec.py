"""Spec-driven harness core (``HARNESS_SPEC.md`` / ADR-012).

Pure, importable logic behind ``scripts/validate.py`` and
``scripts/select_next.py``: feature-catalog loading, JSON-schema validation,
dependency-DAG integrity (dangling edges + cycle detection), git provenance,
per-feature ``validation_command`` execution, and DAG-aware next-feature
selection.

Design notes
------------
* **No side-effecting output.** These functions return data; the CLI shims own
  all presentation. That keeps the harness guarantees unit-testable and
  reusable (mirrors how :mod:`mousedroid.validation.preflight` returns a typed
  report that :mod:`mousedroid.cli.preflight` renders).
* **Dependency-injectable.** :func:`run_features` accepts ``runner`` and
  ``rev_checker`` callables so the orchestration is testable without spawning
  subprocesses.
* **Import-light.** ``jsonschema`` is a *dev-only* dependency and is imported
  lazily, so importing this module never hard-requires it (a missing library
  degrades to a recorded warning, never an import crash).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Feature = dict[str, Any]

#: Tier assigned to a feature that omits ``tier`` (HARNESS_SPEC.md §5).
DEFAULT_TIER = "fast"

#: Allowed execution tiers (mirrors the ``tier`` enum in ``features.schema.json``).
#: The CLI validates ``--tier`` against this so a typo fails loudly instead of
#: silently matching no features and exiting 0.
VALID_TIERS: frozenset[str] = frozenset({"fast", "slow", "hardware"})

#: Lower sorts first — ``select_next`` prefers higher-priority ready features.
PRIORITY: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_features(path: str | Path) -> list[Feature]:
    """Load the ``features`` list from a ``features.yaml`` document.

    Args:
        path: Path to the YAML catalog.

    Returns:
        The list of feature mappings.

    Raises:
        ValueError: If the document is empty, not a mapping, or its
            ``features`` value is missing or not a list.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "features" not in data:
        raise ValueError(f"{path}: missing top-level 'features' list")
    feats = data["features"]
    if not isinstance(feats, list):
        raise ValueError(f"{path}: 'features' must be a list")
    return feats


# --------------------------------------------------------------------------- #
# Structural + DAG checks
# --------------------------------------------------------------------------- #
def check_schema(feats: list[Feature], schema_path: str | Path) -> list[str]:
    """Validate ``feats`` against the JSON Schema at ``schema_path``.

    Args:
        feats: Feature list to validate.
        schema_path: Path to ``features.schema.json``.

    Returns:
        A list of human-readable schema-error strings (empty when valid).

    Raises:
        ModuleNotFoundError: If ``jsonschema`` is not installed. Callers that
            want graceful degradation should catch this (the CLI records a
            warning and skips the structural check).
    """
    import json

    import jsonschema

    with open(schema_path, encoding="utf-8") as fh:
        schema = json.load(fh)
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"schema: {list(e.path)}: {e.message}"
        for e in sorted(validator.iter_errors({"features": feats}), key=lambda e: list(e.path))
    ]


def check_dag(feats: list[Feature]) -> list[str]:
    """Check dependency-DAG integrity: dangling edges and cycles.

    Args:
        feats: Feature list whose ``depends_on`` edges form the DAG.

    Returns:
        A list of error strings — one per malformed entry, dangling edge, and
        detected cycle (empty when the DAG is well-formed). Malformed entries
        (non-mapping items, missing/non-string ``id``, duplicate ``id``) are
        reported rather than raised, so callers that skip schema validation
        (e.g. ``jsonschema`` absent) still get a structured result.
    """
    errs: list[str] = []
    by_id: dict[str, Feature] = {}
    for i, f in enumerate(feats):
        if not isinstance(f, dict):
            errs.append(f"dag: feature[{i}] is not an object")
            continue
        fid = f.get("id")
        if not isinstance(fid, str) or not fid:
            errs.append(f"dag: feature[{i}] missing valid id")
            continue
        if fid in by_id:
            errs.append(f"dag: duplicate feature id {fid}")
            continue
        by_id[fid] = f

    for fid, f in by_id.items():
        for dep in f.get("depends_on", []):
            if dep not in by_id:
                errs.append(f"dag: {fid} depends_on unknown id {dep}")

    white, grey, black = 0, 1, 2
    color = dict.fromkeys(by_id, white)

    def visit(node: str, stack: list[str]) -> None:
        color[node] = grey
        for dep in by_id[node].get("depends_on", []):
            if dep not in by_id:
                continue
            if color[dep] == grey:
                errs.append("dag: cycle " + " -> ".join([*stack, dep]))
            elif color[dep] == white:
                visit(dep, [*stack, dep])
        color[node] = black

    for fid in by_id:
        if color[fid] == white:
            visit(fid, [fid])
    return errs


# --------------------------------------------------------------------------- #
# Provenance + command execution
# --------------------------------------------------------------------------- #
def git_rev_ok(ref: str | None, *, cwd: str | Path | None = None) -> bool:
    """Return whether ``ref`` resolves to a real git commit.

    Args:
        ref: A commit SHA or branch name (``implemented_in``). Falsy values
            return ``False`` without invoking git.
        cwd: Directory to resolve the ref in (defaults to the process CWD).

    Returns:
        ``True`` iff ``git rev-parse`` resolves ``ref`` to a commit.
    """
    if not ref:
        return False
    # Fixed argv list, no shell, trusted constant program (S603/S607 ignored
    # for this file in pyproject.toml, matching validation/runtime.py).
    r = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return r.returncode == 0


def run_validation(f: Feature, *, cwd: str | Path | None = None) -> str | None:
    """Run a feature's ``validation_command`` and report failure.

    Args:
        f: The feature whose command to run.
        cwd: Working directory for the command (defaults to the process CWD).
            The CLI passes the repo root so repo-relative commands resolve
            regardless of where ``validate.py`` was invoked.

    Returns:
        ``None`` on success (exit 0); otherwise an error string carrying the
        exit code and the last few output lines.
    """
    cmd = f.get("validation_command")
    if not cmd:
        return f"{f['id']}: no validation_command defined"
    # shell=True is intentional: validation_command is an operator-authored
    # shell string (HARNESS_SPEC.md §5), not untrusted input. S602 is ignored
    # for this file in pyproject.toml. stderr is merged into stdout so a pytest
    # traceback (which a test runner emits to stdout) is never clobbered by a
    # late stderr warning when the tail is taken.
    r = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd
    )
    if r.returncode != 0:
        tail = r.stdout.strip().splitlines()[-20:]
        return f"{f['id']}: validation_command failed ({r.returncode})\n      " + "\n      ".join(
            tail
        )
    return None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a full ``features.yaml`` validation pass."""

    errors: list[str]
    warnings: list[str]
    ran: int
    skipped: int
    done: int

    @property
    def ok(self) -> bool:
        """``True`` iff no errors were recorded."""
        return not self.errors


def run_features(
    feats: list[Feature],
    schema_path: str | Path | None,
    tiers: set[str],
    *,
    strict_git: bool = False,
    runner: Callable[[Feature], str | None] = run_validation,
    rev_checker: Callable[[str | None], bool] = git_rev_ok,
) -> ValidationResult:
    """Validate structure + DAG, then run ``done`` features in ``tiers``.

    Args:
        feats: The feature catalog.
        schema_path: Path to the JSON Schema, or ``None`` to skip the
            structural check.
        tiers: Tier names whose ``done`` features should be executed.
        strict_git: When ``True``, an unresolved ``implemented_in`` ref is an
            error; otherwise it is a warning.
        runner: Callable that runs a feature and returns an error string or
            ``None`` (injectable for testing).
        rev_checker: Callable that returns whether a ref resolves (injectable
            for testing).

    Returns:
        A :class:`ValidationResult` aggregating errors, warnings, and counts.
        When schema or DAG checks record errors, the function short-circuits
        before running any ``validation_command`` (an invalid catalog must not
        drive command execution); the ``done`` features are reported as
        ``skipped``.

    Notes:
        A requested schema check (``schema_path`` is not ``None``) with
        ``jsonschema`` unavailable is a hard error, not a warning: the harness
        is an enforced gate and must never report a false-green run by silently
        skipping structural validation. Pass ``schema_path=None`` to skip the
        structural check deliberately.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if schema_path is not None:
        try:
            errors += check_schema(feats, schema_path)
        except ModuleNotFoundError:
            errors.append(
                "jsonschema not installed; cannot validate structure "
                "(install the [dev] extra — the harness will not skip the "
                "structural check silently)"
            )
    errors += check_dag(feats)

    if errors:
        done = sum(1 for f in feats if isinstance(f, dict) and f.get("status") == "done")
        return ValidationResult(errors=errors, warnings=warnings, ran=0, skipped=done, done=done)

    ran = skipped = 0
    for f in feats:
        if f.get("status") != "done":
            continue
        if not rev_checker(f.get("implemented_in")):
            msg = (
                f"{f['id']}: implemented_in '{f.get('implemented_in')}' is not a resolvable git ref"
            )
            (errors if strict_git else warnings).append(msg)
        if f.get("tier", DEFAULT_TIER) in tiers:
            err = runner(f)
            ran += 1
            if err:
                errors.append(err)
        else:
            skipped += 1

    done = sum(1 for f in feats if f.get("status") == "done")
    return ValidationResult(errors=errors, warnings=warnings, ran=ran, skipped=skipped, done=done)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Selection:
    """The next action an agent should take, per the dependency DAG.

    ``kind`` is one of ``resume`` (an ``in_progress`` feature to continue),
    ``ready`` (a ``todo`` feature whose deps are met), ``blocked`` (only
    dependency-gated ``todo`` features remain), or ``complete`` (no ``todo``).
    """

    kind: str
    feature: Feature | None = None
    blocked: list[tuple[str, list[str]]] = field(default_factory=list)


def select_next(feats: list[Feature]) -> Selection:
    """Pick the next feature to work on, honouring ``depends_on``.

    Args:
        feats: The feature catalog.

    Returns:
        A :class:`Selection` describing what to do next.
    """
    by_id = {f["id"]: f for f in feats}

    def deps_done(f: Feature) -> bool:
        return all(by_id.get(d, {}).get("status") == "done" for d in f.get("depends_on", []))

    def best(candidates: list[Feature]) -> Feature:
        return sorted(candidates, key=lambda x: (PRIORITY[x["priority"]], x["id"]))[0]

    inprog = [f for f in feats if f.get("status") == "in_progress"]
    if inprog:
        return Selection("resume", feature=best(inprog))

    ready = [f for f in feats if f.get("status") == "todo" and deps_done(f)]
    if ready:
        return Selection("ready", feature=best(ready))

    blocked = [f for f in feats if f.get("status") == "todo" and not deps_done(f)]
    if blocked:
        gated = [
            (f["id"], [d for d in f["depends_on"] if by_id.get(d, {}).get("status") != "done"])
            for f in blocked
        ]
        return Selection("blocked", blocked=gated)

    return Selection("complete")
