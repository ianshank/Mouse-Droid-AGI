#!/usr/bin/env python3
"""Detect module-level concrete-class imports across subsystem boundaries.

Architecture invariant 1 (CLAUDE.md): "Concrete types are only imported
inside factory functions." A peer-reviewed audit found this held in 27 of
32 swept `src/mousedroid` subsystems, with the exceptions falling into two
buckets: (a) a real, previously-uncaught violation (`learning/offline_rl.py`
importing `NoOpExperimentLogger` at module scope — since fixed by moving it
under ``TYPE_CHECKING`` + a function-scoped import, mirroring
`training/pipeline_orchestrator.py`'s existing pattern), and (b) a small set
of already-triaged, deliberately-accepted exceptions (see
``_ALLOWED_CROSS_SUBSYSTEM_IMPORTS`` below).

This scans the WHOLE tree every run (not just a git diff) — unlike
``check_no_hardcoded_values.py``, an import-boundary violation is exactly as
real on line 1 of a file untouched this year as it is on a freshly-added
line, so there is no "existing debt is tolerated" carve-out here.

Scope: only class-valued imports are in scope. The invariant's own wording
("concrete TYPES") is about swappable, stateful implementations — a plain
function or module-level constant (``load_settings``, ``INT16_MAX_F``,
``command_set_supports_lateral``) carries no DI concern and is out of
scope by construction, not via an allowlist.

For a class-valued import, resolve it (dynamic import) and exempt it when
it is a dataclass, a Pydantic ``BaseModel``, an ``Enum``/``IntEnum``/``Flag``,
an ``Exception`` subclass, or a ``typing.Protocol`` — the audit found this
is an established, deliberate codebase-wide convention (DTOs, enums, and
exceptions cross subsystem boundaries freely; only concrete *behavior*-
bearing implementations are meant to stay behind a Protocol + factory
seam). What's left after all of that is a small, explicit, reviewed
allowlist for the remaining genuinely-concrete exceptions the audit judged
legitimate (e.g. training code needing an actual trainable ``nn.Module``).
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pydantic

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "mousedroid"


def _discover_subsystems() -> frozenset[str]:
    """Every direct src/mousedroid/ subdirectory containing at least one .py file.

    Files directly at src/mousedroid/ root (factory.py, constants.py,
    main.py, __main__.py, __init__.py) are the designated composition
    root / package surface and are exempt by construction — this only
    walks subsystem *subdirectories*, never the root itself.
    """
    return frozenset(
        p.name
        for p in _SRC.iterdir()
        if p.is_dir() and p.name != "__pycache__" and any(p.rglob("*.py"))
    )


# Prefixes that are always a safe dependency for any subsystem — the
# project's documented shared kernel (config, logging, constants, common,
# utils), plus resilience.circuit_breaker/retry (audited as matching the
# shared-kernel *spirit*: stateless, config-driven, never swapped, never
# factory-wired — every current consumer, including resilience's own
# internal wrappers, constructs it identically).
_SHARED_KERNEL_PREFIXES = (
    "mousedroid.config.schema",
    "mousedroid.config.loader",
    "mousedroid.logging.setup",
    "mousedroid.constants",
    "mousedroid.common.",
    "mousedroid.common",
    "mousedroid.utils.",
    "mousedroid.utils",
    "mousedroid.resilience.circuit_breaker",
    "mousedroid.resilience.retry",
)

# Already-triaged, deliberately-accepted concrete-class cross-subsystem
# imports that don't fit the dataclass/enum/exception/Protocol exemption —
# each entry is (importing file, imported module, imported name). Adding a
# new entry requires the same documented-reason bar as the noqa/type:ignore
# suppression budgets: this is a ratchet, not a bypass valve.
_ALLOWED_CROSS_SUBSYSTEM_IMPORTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        # Training-time code needs the actual trainable nn.Module — there is
        # no training-time surface on WorldModelProtocol (inference-only:
        # observe_step) to depend on instead. Audited, judged legitimate.
        (
            "src/mousedroid/training/drift_reduction.py",
            "mousedroid.world_model.rssm",
            "RSSM",
        ),
        (
            "src/mousedroid/training/drift_reduction.py",
            "mousedroid.world_model.rssm",
            "DriftCorrectionHead",
        ),
        (
            "src/mousedroid/training/drift_reduction.py",
            "mousedroid.world_model.rssm",
            "RawModalityDecoders",
        ),
        # Dormant, currently-safe duplication of factory.py's
        # build_injection_filter resolution logic (self-defaulting when
        # constructed outside the factory). Audited: documented + intentional
        # (three independent docstrings), unreachable via the only
        # production construction path, no divergence under default config.
        # Tracked as a hardening opportunity, not fixed in this pass — see
        # the code-hygiene plan's "Deferred findings".
        (
            "src/mousedroid/llm_gateway/gateway.py",
            "mousedroid.security.injection_filter",
            "RegexInjectionFilter",
        ),
        (
            "src/mousedroid/llm_gateway/anthropic_gateway.py",
            "mousedroid.security.injection_filter",
            "RegexInjectionFilter",
        ),
        # Same-shape null-object default as RealClock (module-level, same
        # file, unflagged) — audited: no import-cycle risk, no established
        # need for a deferred import here (unlike orchestrator.py's
        # NullHookRegistry/NullJournal, which defer specifically to avoid a
        # real cycle). Peer-reviewed conclusion: not worth touching absent a
        # concrete reason, since deferring would be purely cosmetic.
        (
            "src/mousedroid/voice/rocky.py",
            "mousedroid.telemetry.failure_recorder",
            "NullFailureRecorder",
        ),
    }
)


@dataclass(frozen=True)
class Violation:
    """A single flagged cross-subsystem module-level class import."""

    file: str
    line: int
    module: str
    names: tuple[str, ...]


def _subsystem_of(path: Path) -> str:
    return path.relative_to(_SRC).parts[0]


def _is_type_checking_test(test: ast.expr) -> bool:
    """Match ``if TYPE_CHECKING:`` in either ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING`` form."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _is_shared_kernel(module: str) -> bool:
    return any(module == prefix or module.startswith(prefix) for prefix in _SHARED_KERNEL_PREFIXES)


def _is_protocol_module(module: str) -> bool:
    parts = module.split(".")
    return any(part in ("protocol", "protocols") for part in parts)


def _module_level_mousedroid_imports(tree: ast.Module) -> list[ast.ImportFrom]:
    """Top-level ``from mousedroid.X import ...`` statements only.

    Deliberately does NOT recurse into function/method/class bodies
    (deferred imports are exempt by design) and skips the bodies of
    ``if TYPE_CHECKING:`` blocks (those never execute).
    """
    found: list[ast.ImportFrom] = []
    for node in tree.body:
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("mousedroid.")
        ):
            found.append(node)
    return found


def _resolve(module: str, name: str) -> object | None:
    try:
        mod = importlib.import_module(module)
        resolved: object = getattr(mod, name)
        return resolved
    except Exception as exc:
        print(f"  (could not resolve {module}.{name}: {exc})", file=sys.stderr)
        return None


def _is_dto_enum_exception_or_protocol(obj: type) -> bool:
    if dataclasses.is_dataclass(obj):
        return True
    if issubclass(obj, pydantic.BaseModel):
        return True
    if issubclass(obj, enum.Enum):
        return True
    if issubclass(obj, BaseException):
        return True
    return bool(getattr(obj, "_is_protocol", False))


def _needs_review(module: str, name: str) -> bool:
    """True when this class-valued import is a genuine concrete-type violation.

    A resolution failure fails CLOSED (treated as needing review) — an
    unresolvable import is exactly the kind of thing a human should look
    at, not silently wave through.
    """
    obj = _resolve(module, name)
    if obj is None:
        return True
    if not isinstance(obj, type):
        return False  # functions/constants are not "concrete types"
    return not _is_dto_enum_exception_or_protocol(obj)


def _check_file(path: Path, subsystems: frozenset[str]) -> list[Violation]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        print(f"Failed to parse {path}: {exc}", file=sys.stderr)
        return []

    own_subsystem = _subsystem_of(path)
    rel_path = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[Violation] = []

    for node in _module_level_mousedroid_imports(tree):
        module = node.module or ""
        if _is_shared_kernel(module) or _is_protocol_module(module):
            continue

        module_parts = module.split(".")
        # module_parts[0] == "mousedroid"; module_parts[1] is the subsystem
        # when present (e.g. "mousedroid.learning.offline_rl" -> "learning").
        target_subsystem = module_parts[1] if len(module_parts) > 1 else None
        if target_subsystem is None or target_subsystem not in subsystems:
            continue  # not a subsystem-scoped import (e.g. mousedroid.factory itself)
        if target_subsystem == own_subsystem:
            continue  # intra-subsystem import, always fine

        flagged_names = []
        for alias in node.names:
            name = alias.name
            if name.endswith("Protocol"):
                continue  # codebase-wide naming convention, cheap fast-path
            if (rel_path, module, name) in _ALLOWED_CROSS_SUBSYSTEM_IMPORTS:
                continue
            if _needs_review(module, name):
                flagged_names.append(name)

        if flagged_names:
            violations.append(Violation(rel_path, node.lineno, module, tuple(flagged_names)))

    return violations


def find_violations() -> list[Violation]:
    subsystems = _discover_subsystems()
    violations: list[Violation] = []
    for subsystem in sorted(subsystems):
        for path in sorted((_SRC / subsystem).rglob("*.py")):
            violations.extend(_check_file(path, subsystems))
    return violations


def main() -> int:
    violations = find_violations()
    if not violations:
        print(
            "Subsystem-boundary gate passed: no module-level cross-subsystem concrete-class imports."
        )
        return 0

    print("Module-level cross-subsystem concrete-class imports detected:")
    for v in violations:
        names = ", ".join(v.names)
        print(f"- {v.file}:{v.line} -> from {v.module} import {names}")
    print(
        "\nProtocol-based DI (invariant 1): a subsystem may depend on another "
        "subsystem's Protocol, a DTO/enum/exception, the shared kernel, or its "
        "own package — never another subsystem's concrete, behavior-bearing "
        "implementation at module scope. Defer the import to function scope "
        "(see learning/offline_rl.py or training/pipeline_orchestrator.py for "
        "the established pattern), or add a documented, reviewed entry to "
        "_ALLOWED_CROSS_SUBSYSTEM_IMPORTS in scripts/check_subsystem_boundaries.py "
        "if the coupling is genuinely unavoidable (e.g. training code needing a "
        "concrete trainable model)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
