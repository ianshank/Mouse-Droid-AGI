"""Repository-root resolution and glob matching for workforce tooling.

Portability contract (workforce invariant I-3): nothing here — and nothing that
calls here — may embed an absolute filesystem path or the repository name. The
root is discovered at runtime from ``$CLAUDE_PROJECT_DIR`` (set by Claude Code
for hook commands), then by walking up for repository marker files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Environment variable Claude Code exports for hook commands.
PROJECT_DIR_ENV = "CLAUDE_PROJECT_DIR"

#: Markers identifying a repository root, strongest first. A directory must
#: contain *all* entries in a group to qualify.
#:
#: Group order is significant and is applied group-major (every ancestor is
#: tested against the strongest group before the next group is tried). Walking
#: ancestor-major instead would let a vendored subproject's lone
#: ``pyproject.toml`` shadow the real root, because the walk starts at the
#: deepest directory — the bare-``pyproject.toml`` group is a last-resort
#: fallback for checkouts without ``.git`` (a tarball export), not a peer.
_ROOT_MARKER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("pyproject.toml", ".git"),
    ("pyproject.toml", "features.yaml"),
    ("pyproject.toml",),
)

# Glob translation: ``**`` crosses separators, ``*``/``?`` do not.
_GLOB_TOKEN_RE = re.compile(r"\*\*/|\*\*|\*|\?")
_WILDCARD_CHARS = frozenset("*?[")


def resolve_repo_root(start: Path | None = None, *, env: dict[str, str] | None = None) -> Path:
    """Return the repository root directory.

    Resolution order: ``$CLAUDE_PROJECT_DIR`` (when set and existing), then a
    marker-file walk upward from ``start``, then ``start`` itself as a
    last-resort fallback so callers always receive a usable directory.

    Args:
        start: Directory to begin the upward walk from. Defaults to this
            module's location, which keeps resolution independent of the
            caller's working directory.
        env: Environment mapping to read. Defaults to :data:`os.environ`.

    Returns:
        The resolved repository root.
    """
    environ = os.environ if env is None else env
    configured = environ.get(PROJECT_DIR_ENV, "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    origin = (Path(__file__).resolve().parent if start is None else Path(start)).resolve()
    search_base = origin if origin.is_dir() else origin.parent
    candidates = (search_base, *search_base.parents)
    # Group-major: exhaust every ancestor against a stronger marker group before
    # falling back to a weaker one, so a nested `pyproject.toml` cannot shadow
    # the real root.
    for group in _ROOT_MARKER_GROUPS:
        for directory in candidates:
            if all((directory / marker).exists() for marker in group):
                return directory
    return search_base


def to_repo_relative(path: str | Path, repo_root: Path) -> str | None:
    """Return ``path`` as a repo-relative POSIX string, or ``None`` if outside.

    Args:
        path: Absolute or relative filesystem path.
        repo_root: The repository root to relativise against.

    Returns:
        The POSIX-style relative path, or ``None`` when ``path`` lies outside
        ``repo_root``.

    Note:
        Normalisation is **lexical** (:func:`os.path.normpath`), not
        filesystem-resolving: ``Path.resolve()`` is deliberately avoided so a
        pending, not-yet-written edit target still maps to a repo-relative
        path. The tradeoff is that a symlink pointing outside the repository is
        not detected here — the freeze gate is an edit-time guardrail against
        mistakes, not a sandbox against a determined bypass (an operator who
        can plant a symlink can also set the override env var).
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = Path(os.path.normpath(str(candidate)))
    root = Path(os.path.normpath(str(repo_root.expanduser())))
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return None


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a POSIX-style glob into an anchored regex.

    Semantics (git/gitignore-like rather than :mod:`fnmatch`-like, whose ``*``
    wrongly crosses directory separators):

    * ``**`` matches across separators;
    * ``*`` and ``?`` never cross a separator;
    * a pattern with no wildcard matches the path exactly *or* as a directory
      prefix, so ``src/pkg/sub`` covers ``src/pkg/sub/file.py``.

    Args:
        pattern: The glob pattern.

    Returns:
        A compiled, fully anchored regular expression.
    """
    normalised = pattern.strip().replace("\\", "/")
    if not any(char in normalised for char in _WILDCARD_CHARS):
        literal = re.escape(normalised.rstrip("/"))
        return re.compile(rf"^{literal}(?:/.*)?$")

    out: list[str] = []
    index = 0
    for match in _GLOB_TOKEN_RE.finditer(normalised):
        out.append(re.escape(normalised[index : match.start()]))
        piece = match.group(0)
        if piece == "**/":
            # Match zero or more leading directories.
            out.append("(?:.*/)?")
        elif piece == "**":
            out.append(".*")
        elif piece == "*":
            out.append("[^/]*")
        else:  # "?"
            out.append("[^/]")
        index = match.end()
    out.append(re.escape(normalised[index:]))
    return re.compile(rf"^{''.join(out)}$")


def path_matches_any(rel_path: str, patterns: list[str]) -> str | None:
    """Return the first pattern in ``patterns`` matching ``rel_path``.

    Args:
        rel_path: Repo-relative POSIX path.
        patterns: Glob patterns, evaluated in order.

    Returns:
        The matching pattern, or ``None`` when nothing matches.
    """
    normalised = rel_path.replace("\\", "/").lstrip("./")
    for pattern in patterns:
        if not pattern.strip():
            continue
        if glob_to_regex(pattern).match(normalised):
            return pattern
    return None
