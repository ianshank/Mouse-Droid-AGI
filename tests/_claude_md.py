"""Shared helpers for reasoning about Claude Code instruction files.

Follows the repo's ``tests/_<name>.py`` shared-helper convention (cf.
``tests/_bash.py``, ``tests/_pyproject.py``, ``tests/_script_loader.py``).

**Why the import detection lives here rather than inline in a test.** Claude Code
expands ``@path`` imports in a ``CLAUDE.md``, but *"import parsing skips Markdown
code spans and fenced code blocks"* — so ``@AGENTS.md`` is an import and
``` `@AGENTS.md` ``` is not. A test that greps for the bare substring passes on a
file that only *mentions* the path, which is precisely the failure mode F-053
exists to prevent: a documentation file that nothing loads, guarded by a check
that cannot tell the difference.

That distinction is one regex with two exclusions, it is easy to get wrong, and
it is needed by more than one caller, so it is named once here.

**Why the file sets are discovered, not enumerated.** ``AGENTS.md`` and
``CLAUDE.md`` locations change as packages are added. A hardcoded roster silently
stops covering new directories, which is the same defect one level up.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, get_origin

__all__ = [
    "AGENTS_MD",
    "AGENT_FACING_WHEEL_PATTERNS",
    "CLAUDE_MD",
    "KEY_FILES_EXEMPTION_LAPSE_FEATURE",
    "KEY_FILES_PATH_EXEMPTIONS",
    "ConfigValueClaim",
    "NamedPathClaim",
    "discover_in_package_doc_packages",
    "discover_nested_claude_md",
    "discover_tracked",
    "feature_status",
    "illegal_config_value_claims",
    "imported_paths",
    "imports_target",
    "iter_config_value_claims",
    "iter_key_files_claims",
    "iter_named_py_path_claims",
    "key_files_section",
    "repo_root",
    "surface_map_indexes",
    "unresolved_path_claims",
]

AGENTS_MD = "AGENTS.md"
CLAUDE_MD = "CLAUDE.md"

#: Filename globs that must stay out of the built wheel. Agent instructions are
#: internal; ``pyproject.toml``'s own comment records that without the exclusion
#: the wheel ships them to PyPI. Kept here so the wheel gate and any future
#: agent-facing filename share one list instead of drifting apart.
AGENT_FACING_WHEEL_PATTERNS: tuple[str, ...] = (
    "**/CLAUDE.md",
    "**/agent.md",
    "**/AGENTS.md",
)

# A fenced block, then an inline code span. Stripped in that order so a fence
# containing backticks cannot leave a stray span behind.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_SPAN_RE = re.compile(r"`[^`\n]*`")

# Claude Code's import form: '@' followed by a path, at a word boundary. Matches
# the documented syntax (relative or absolute, resolved against the importing
# file) without trying to validate the path itself.
_IMPORT_RE = re.compile(r"(?<![\w`])@([A-Za-z0-9_./~@-]+)")


def repo_root() -> Path:
    """Repository root, resolved from this file rather than the cwd."""
    return Path(__file__).resolve().parents[1]


def _strip_code(text: str) -> str:
    """Remove fenced blocks and inline spans, where ``@path`` is not an import."""
    return _SPAN_RE.sub(" ", _FENCE_RE.sub(" ", text))


def imported_paths(markdown: str) -> tuple[str, ...]:
    """Every ``@path`` import in ``markdown``, in order, code excluded.

    Args:
        markdown: Raw file contents.

    Returns:
        The imported path strings. A path mentioned only inside a code span or a
        fenced block is absent, because Claude Code does not import those.
    """
    return tuple(_IMPORT_RE.findall(_strip_code(markdown)))


def imports_target(markdown: str, target: str) -> bool:
    """Whether ``markdown`` imports ``target`` as a real (non-code) import."""
    return target in imported_paths(markdown)


def discover_tracked(filename: str) -> tuple[Path, ...]:
    """Every git-tracked file named ``filename``, as repo-relative paths.

    Uses ``git ls-files`` rather than ``rglob`` so untracked scratch files and
    ignored trees cannot influence a gate. Sorted for deterministic failure
    messages.

    Args:
        filename: Exact basename to match, case-sensitively.

    Returns:
        Repo-relative paths, sorted.
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", f"*{filename}", filename],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root(),
    )
    return tuple(
        sorted(
            Path(entry)
            for entry in completed.stdout.split("\0")
            if entry and Path(entry).name == filename
        )
    )


def discover_in_package_doc_packages() -> tuple[str, ...]:
    """Package names under ``src/mousedroid/`` that carry in-package agent docs.

    A package "has in-package docs" when git tracks an ``agent.md`` and/or a
    ``CLAUDE.md`` directly inside ``src/mousedroid/<package>/``. Discovered
    rather than enumerated so a new subsystem that lands either file is
    covered without editing a roster.

    Returns:
        Sorted package directory names (not paths).
    """
    packages: set[str] = set()
    for filename in (CLAUDE_MD, "agent.md"):
        for path in discover_tracked(filename):
            parts = path.parts
            if (
                len(parts) == 4
                and parts[0] == "src"
                and parts[1] == "mousedroid"
                and parts[3] == filename
            ):
                packages.add(parts[2])
    return tuple(sorted(packages))


def surface_map_indexes(root_claude_md: str, package: str) -> bool:
    """Whether the root Surface Map links to ``src/mousedroid/<package>/CLAUDE.md``.

    Matches the ``file:///src/mousedroid/...`` link form used by the root
    ``CLAUDE.md`` Surface Map. Code spans and fences are excluded so a
    backticked mention cannot satisfy the gate.
    """
    needle = f"file:///src/mousedroid/{package}/{CLAUDE_MD}"
    return needle in _strip_code(root_claude_md)


# ---------------------------------------------------------------------------
# Nested CLAUDE.md Key Files / named-path / config-value gates (F-053 Phase 3)
# ---------------------------------------------------------------------------
#
# Generalises the orchestrator-only symbol pin in
# ``tests/regression/test_doc_reconciliation_aqa.py`` across every nested
# ``CLAUDE.md``. Paths and ``file::Symbol`` entries must resolve; config values
# named next to ``SomeConfig.field`` must be legal ``Literal`` members.

_KEY_FILES_SECTION_RE = re.compile(
    r"^## Key Files[^\n]*\n(.*?)(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_FILE_SYMBOL_RE = re.compile(r"^((?:[\w./~+-]+/)?[\w.-]+\.py)::(\w+)$")
_FILE_PATH_RE = re.compile(r"^((?:[\w./~+-]+/)?[\w.-]+\.py)$")
_DIR_PATH_RE = re.compile(r"^((?:[\w./~+-]+/)+)$")
_CONFIG_FIELD_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+Config)\.([a-z][a-z0-9_]*)\b")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FUNCTION_STYLE = re.compile(r"^(?:get_|configure_|safe_|compute_|build_|load_)[a-z0-9_]+$")

# Lines that name a path only to deny or historicise it. Treating those as
# positive claims would force the doc to stop saying what is absent.
_NEGATIVE_OR_HISTORICAL_PATH = re.compile(
    r"\b(?:there is no|no longer|split from|former|monolithic|does not exist|"
    r"do not exist|removed|renamed)\b",
    re.IGNORECASE,
)

#: Nested ``CLAUDE.md`` files exempt from the named-path gate. Exact
#: repo-relative paths only (never a prefix) — same discipline as
#: ``_GATED_FACTORY_FILES`` in ``tests/regression/test_f042_aqa.py``.
#: ``src/mousedroid/arm/**`` is frozen by ``freeze_gate`` while F-008 is
#: ``todo``; the stale ``mock_arm.py`` claim cannot be edited until then.
KEY_FILES_PATH_EXEMPTIONS: frozenset[str] = frozenset(
    {
        "src/mousedroid/arm/CLAUDE.md",
    }
)

#: Feature whose ``status: done`` lapses :data:`KEY_FILES_PATH_EXEMPTIONS`.
KEY_FILES_EXEMPTION_LAPSE_FEATURE = "F-008"


@dataclass(frozen=True, slots=True)
class NamedPathClaim:
    """A backticked path or ``file::Symbol`` claim inside a nested ``CLAUDE.md``."""

    doc: Path
    lineno: int
    raw: str
    kind: str  # "file" | "dir" | "symbol"
    path: str
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigValueClaim:
    """A prose claim that a config field may take a named value."""

    doc: Path
    lineno: int
    config_class: str
    field: str
    claimed_value: str


def discover_nested_claude_md() -> tuple[Path, ...]:
    """Every tracked ``src/mousedroid/<pkg>/CLAUDE.md``, as repo-relative paths."""
    return tuple(
        path
        for path in discover_tracked(CLAUDE_MD)
        if (
            len(path.parts) == 4
            and path.parts[0] == "src"
            and path.parts[1] == "mousedroid"
            and path.parts[3] == CLAUDE_MD
        )
    )


def key_files_section(markdown: str) -> str | None:
    """Body of the ``## Key Files`` section, or ``None`` if absent."""
    match = _KEY_FILES_SECTION_RE.search(markdown)
    return match.group(1) if match else None


def _resolve_under(package_dir: Path, rel: str, root: Path) -> Path | None:
    """Resolve ``rel`` against the package, ``src/mousedroid/``, then the repo root."""
    candidates = (
        package_dir / rel,
        root / "src" / "mousedroid" / rel,
        root / rel,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    name = Path(rel).name
    if name and "/" not in rel.rstrip("/"):
        hits = sorted(package_dir.rglob(name))
        if hits:
            return min(hits, key=lambda p: len(p.relative_to(package_dir).parts))
        repo_hits = [
            p for p in root.rglob(name) if ".git" not in p.parts and ".venv" not in p.parts
        ]
        if len(repo_hits) == 1:
            return repo_hits[0]
    return None


def _definitions_in(path: Path) -> set[str]:
    """Class and function names defined at any scope in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _primary_spans_before_emdash(line: str) -> list[str]:
    """Backtick spans on a Key Files bullet before the description separator."""
    cut = line
    for sep in (" — ", " -- "):
        if sep in line:
            cut = line.split(sep, 1)[0]
            break
    return _BACKTICK_RE.findall(cut)


def _claim_from_primary_span(
    rel_doc: Path,
    lineno: int,
    span: str,
    current_file: str | None,
) -> tuple[NamedPathClaim | None, str | None, str | None]:
    """Parse one Key Files primary span into a claim and updated file/dir cursors.

    Returns ``(claim_or_None, new_current_file, new_current_dir_or_None)``.
    ``new_current_dir_or_None`` is only set when the span is a directory; callers
    must otherwise keep their existing ``current_dir``.
    """
    file_sym = _FILE_SYMBOL_RE.fullmatch(span)
    if file_sym:
        path, symbol = file_sym.group(1), file_sym.group(2)
        return (
            NamedPathClaim(rel_doc, lineno, span, "symbol", path, symbol),
            path,
            None,
        )
    file_only = _FILE_PATH_RE.fullmatch(span)
    if file_only:
        path = file_only.group(1)
        return NamedPathClaim(rel_doc, lineno, span, "file", path), path, None
    dir_only = _DIR_PATH_RE.fullmatch(span)
    if dir_only:
        path = dir_only.group(1)
        return NamedPathClaim(rel_doc, lineno, span, "dir", path), current_file, path
    if (
        current_file
        and _IDENTIFIER_RE.fullmatch(span)
        and (span[0].isupper() or _FUNCTION_STYLE.fullmatch(span))
    ):
        return (
            NamedPathClaim(rel_doc, lineno, span, "symbol", current_file, span),
            current_file,
            None,
        )
    return None, current_file, None


def _parenthetical_file_claims(
    *,
    rel_doc: Path,
    lineno: int,
    line: str,
    prev: str,
    current_dir: str | None,
    existing: list[NamedPathClaim],
    package_dir: Path,
) -> list[NamedPathClaim]:
    """``*.py`` basenames in a Key Files description, directory-qualified when real."""
    if _NEGATIVE_OR_HISTORICAL_PATH.search(line) or _NEGATIVE_OR_HISTORICAL_PATH.search(prev):
        return []
    found: list[NamedPathClaim] = []
    for span in _BACKTICK_RE.findall(line):
        file_only = _FILE_PATH_RE.fullmatch(span)
        if not file_only:
            continue
        path = file_only.group(1)
        already = any(
            c.lineno == lineno and c.path.endswith(path) and c.kind == "file" for c in existing
        )
        if already:
            continue
        # Prefer directory-qualified path when the bullet named a directory,
        # but only when that join actually exists — otherwise keep the bare
        # name so resolution can still succeed, and so a historical basename
        # on a continuation line is not invented as ``metrics/metrics.py``.
        if current_dir and "/" not in path:
            joined = f"{current_dir}{path}"
            if (package_dir / joined).exists():
                path = joined
        found.append(NamedPathClaim(rel_doc, lineno, span, "file", path))
    return found


def iter_key_files_claims(
    doc: Path, markdown: str, *, root: Path | None = None
) -> list[NamedPathClaim]:
    """Path and ``file::Symbol`` claims from a nested ``CLAUDE.md`` Key Files section."""
    root = root or repo_root()
    section = key_files_section(markdown)
    if section is None:
        return []
    section_match = _KEY_FILES_SECTION_RE.search(markdown)
    assert section_match is not None
    body_start_line = markdown[: section_match.start()].count("\n") + 2
    rel_doc = doc if not doc.is_absolute() else doc.relative_to(root)
    package_dir = root / rel_doc.parent
    lines = section.splitlines()

    claims: list[NamedPathClaim] = []
    current_file: str | None = None
    current_dir: str | None = None
    for offset, line in enumerate(lines):
        lineno = body_start_line + offset
        if line.strip().startswith("-"):
            current_file = None
            current_dir = None
            for span in _primary_spans_before_emdash(line):
                claim, current_file, maybe_dir = _claim_from_primary_span(
                    rel_doc, lineno, span, current_file
                )
                if maybe_dir is not None:
                    current_dir = maybe_dir
                if claim is not None:
                    claims.append(claim)
        prev = lines[offset - 1] if offset > 0 else ""
        claims.extend(
            _parenthetical_file_claims(
                rel_doc=rel_doc,
                lineno=lineno,
                line=line,
                prev=prev,
                current_dir=current_dir,
                existing=claims,
                package_dir=package_dir,
            )
        )
    return claims


def iter_named_py_path_claims(
    doc: Path, markdown: str, *, root: Path | None = None
) -> list[NamedPathClaim]:
    """Every backticked ``*.py`` path in a nested ``CLAUDE.md``, excluding negations."""
    root = root or repo_root()
    rel_doc = doc if not doc.is_absolute() else doc.relative_to(root)
    claims: list[NamedPathClaim] = []
    for lineno, line in enumerate(markdown.splitlines(), start=1):
        if _NEGATIVE_OR_HISTORICAL_PATH.search(line):
            continue
        for span in _BACKTICK_RE.findall(line):
            file_sym = _FILE_SYMBOL_RE.fullmatch(span)
            if file_sym:
                claims.append(NamedPathClaim(rel_doc, lineno, span, "file", file_sym.group(1)))
                continue
            file_only = _FILE_PATH_RE.fullmatch(span)
            if file_only:
                claims.append(NamedPathClaim(rel_doc, lineno, span, "file", file_only.group(1)))
    return claims


def unresolved_path_claims(
    claims: Sequence[NamedPathClaim], *, root: Path | None = None
) -> list[str]:
    """Human-readable unresolved path / symbol claims."""
    root = root or repo_root()
    offenders: list[str] = []
    for claim in claims:
        package_dir = root / claim.doc.parent
        resolved = _resolve_under(package_dir, claim.path, root)
        if claim.kind in {"file", "dir"}:
            if resolved is None:
                offenders.append(
                    f"{claim.doc.as_posix()}:{claim.lineno}: "
                    f"{claim.path!r} does not resolve under "
                    f"{claim.doc.parent.as_posix()}/"
                )
            continue
        if resolved is None:
            offenders.append(
                f"{claim.doc.as_posix()}:{claim.lineno}: "
                f"{claim.path!r} (for symbol {claim.symbol}) does not resolve"
            )
            continue
        assert claim.symbol is not None
        if claim.symbol not in _definitions_in(resolved):
            offenders.append(
                f"{claim.doc.as_posix()}:{claim.lineno}: "
                f"{claim.symbol!r} is not a class/def in "
                f"{resolved.relative_to(root).as_posix()}"
            )
    return offenders


def iter_config_value_claims(
    doc: Path, markdown: str, *, root: Path | None = None
) -> list[ConfigValueClaim]:
    """``SomeConfig.field`` prose claims with nearby lowercase backticked values."""
    root = root or repo_root()
    rel_doc = doc if not doc.is_absolute() else doc.relative_to(root)
    claims: list[ConfigValueClaim] = []
    for lineno, line in enumerate(markdown.splitlines(), start=1):
        fields = list(_CONFIG_FIELD_RE.finditer(line))
        if not fields:
            continue
        # Collect spans with their start offsets so a match that begins *inside*
        # a backtick (`` `LLMConfig.field` ``) does not break pair matching.
        spans = [(m.start(), m.group(1)) for m in _BACKTICK_RE.finditer(line)]
        for match in fields:
            config_class, field = match.group(1), match.group(2)
            for start, span in spans:
                if start < match.start() and span != f"{config_class}.{field}":
                    # Span entirely before the field reference — not "nearby".
                    # Keep the span that *is* the field reference itself only to
                    # skip it; values we want sit at or after the field.
                    continue
                if span in {f"{config_class}.{field}", config_class, field}:
                    continue
                if not re.fullmatch(r"[a-z][a-z0-9_]*", span):
                    continue
                claims.append(ConfigValueClaim(rel_doc, lineno, config_class, field, span))
    return claims


def _literal_values_for_config_field(config_class: str, field: str) -> frozenset[str] | None:
    """Allowed string ``Literal`` members for ``Config.field``, or ``None``."""
    from mousedroid.config.schema import Settings

    models: list[type] = [Settings]
    seen: set[type] = set()
    target: type | None = None
    while models:
        model = models.pop()
        if model in seen:
            continue
        seen.add(model)
        if model.__name__ == config_class:
            target = model
            break
        for info in model.model_fields.values():
            ann = info.annotation
            origin = get_origin(ann)
            args = get_args(ann) if origin is not None else ()
            for cand in (ann, *args):
                if isinstance(cand, type) and hasattr(cand, "model_fields"):
                    models.append(cand)
    if target is None or field not in target.model_fields:
        return None
    annotation = target.model_fields[field].annotation
    values = get_args(annotation)
    if values and all(isinstance(v, str) for v in values):
        return frozenset(values)
    return None


def illegal_config_value_claims(claims: Sequence[ConfigValueClaim]) -> list[str]:
    """Claims whose value is not in the field's ``Literal`` set."""
    offenders: list[str] = []
    for claim in claims:
        allowed = _literal_values_for_config_field(claim.config_class, claim.field)
        if allowed is None:
            continue
        if claim.claimed_value not in allowed:
            offenders.append(
                f"{claim.doc.as_posix()}:{claim.lineno}: "
                f"{claim.config_class}.{claim.field} claims {claim.claimed_value!r}, "
                f"allowed={sorted(allowed)}"
            )
    return offenders


def feature_status(feature_id: str, *, root: Path | None = None) -> str | None:
    """Return the ``status`` for ``feature_id`` from ``features.yaml``."""
    import yaml

    root = root or repo_root()
    data = yaml.safe_load((root / "features.yaml").read_text(encoding="utf-8"))
    for feature in data.get("features", []):
        if feature.get("id") == feature_id:
            status = feature.get("status")
            return str(status) if status is not None else None
    return None
