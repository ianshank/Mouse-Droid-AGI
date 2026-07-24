# tests/unit/tools/claude_hooks/test_paths.py
"""Unit tests for repo-root resolution and glob matching.

The glob semantics are load-bearing: the freeze gate decides what is frozen from
these patterns, so ``*`` must not silently cross a directory separator (the
:mod:`fnmatch` behaviour this module deliberately avoids).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.claude_hooks.paths import (
    PROJECT_DIR_ENV,
    glob_to_regex,
    path_matches_any,
    resolve_repo_root,
    to_repo_relative,
)


def _make_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / ".git").mkdir(exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# resolve_repo_root
# ---------------------------------------------------------------------------


def test_env_var_wins_when_directory_exists(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    resolved = resolve_repo_root(env={PROJECT_DIR_ENV: str(repo)})
    assert resolved == repo.resolve()


def test_env_var_ignored_when_directory_missing(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    resolved = resolve_repo_root(start=nested, env={PROJECT_DIR_ENV: str(tmp_path / "nope")})
    assert resolved == repo.resolve()


def test_blank_env_var_falls_back_to_marker_walk(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    nested = repo / "a" / "b"
    nested.mkdir(parents=True)
    assert resolve_repo_root(start=nested, env={PROJECT_DIR_ENV: "   "}) == repo.resolve()


def test_marker_walk_finds_root_from_nested_dir(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    nested = repo / "deep" / "deeper"
    nested.mkdir(parents=True)
    assert resolve_repo_root(start=nested, env={}) == repo.resolve()


def test_features_yaml_marker_group_matches(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "features.yaml").write_text("features: []\n", encoding="utf-8")
    assert resolve_repo_root(start=tmp_path, env={}) == tmp_path.resolve()


def test_file_start_uses_parent_directory(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    probe = repo / "module.py"
    probe.write_text("x = 1\n", encoding="utf-8")
    assert resolve_repo_root(start=probe, env={}) == repo.resolve()


def test_fallback_returns_start_when_no_marker(tmp_path: Path) -> None:
    lonely = tmp_path / "no_markers"
    lonely.mkdir()
    assert resolve_repo_root(start=lonely, env={}) == lonely.resolve()


def test_default_start_resolves_this_repository() -> None:
    # No args: must find this repository from the module's own location.
    assert (resolve_repo_root(env={}) / "pyproject.toml").is_file()


# ---------------------------------------------------------------------------
# to_repo_relative
# ---------------------------------------------------------------------------


def test_absolute_path_inside_repo(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert to_repo_relative(repo / "src" / "a.py", repo) == "src/a.py"


def test_relative_path_is_resolved_against_root(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert to_repo_relative("src/a.py", repo) == "src/a.py"


def test_path_outside_repo_returns_none(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "repo")
    outside = tmp_path / "elsewhere" / "a.py"
    assert to_repo_relative(outside, repo) is None


def test_nonexistent_path_still_resolves(tmp_path: Path) -> None:
    # A pending write targets a file that does not exist yet.
    repo = _make_repo(tmp_path)
    assert to_repo_relative(repo / "new" / "file.py", repo) == "new/file.py"


def test_dot_segments_are_normalised(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert to_repo_relative(repo / "src" / ".." / "src" / "a.py", repo) == "src/a.py"


# ---------------------------------------------------------------------------
# glob_to_regex / path_matches_any
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "candidate", "expected"),
    [
        ("src/pkg/arm/**", "src/pkg/arm/driver.py", True),
        ("src/pkg/arm/**", "src/pkg/arm/deep/nested/driver.py", True),
        ("src/pkg/arm/**", "src/pkg/armour.py", False),
        ("src/pkg/arm/**", "src/pkg/other/driver.py", False),
        # Single star must NOT cross a separator.
        ("src/*.py", "src/a.py", True),
        ("src/*.py", "src/sub/a.py", False),
        ("**/tests/*.py", "a/b/tests/x.py", True),
        ("**/tests/*.py", "tests/x.py", True),
        ("?.py", "a.py", True),
        ("?.py", "ab.py", False),
        # Wildcard-free patterns match exactly or as a directory prefix.
        ("src/pkg/arm", "src/pkg/arm", True),
        ("src/pkg/arm", "src/pkg/arm/driver.py", True),
        ("src/pkg/arm", "src/pkg/armour.py", False),
        ("src/pkg/arm/", "src/pkg/arm/driver.py", True),
    ],
)
def test_glob_semantics(pattern: str, candidate: str, expected: bool) -> None:
    assert bool(glob_to_regex(pattern).match(candidate)) is expected


def test_path_matches_any_returns_first_matching_pattern() -> None:
    patterns = ["docs/**", "src/pkg/arm/**"]
    assert path_matches_any("src/pkg/arm/x.py", patterns) == "src/pkg/arm/**"


def test_path_matches_any_returns_none_when_nothing_matches() -> None:
    assert path_matches_any("src/other.py", ["src/pkg/arm/**"]) is None


def test_path_matches_any_with_empty_pattern_list() -> None:
    assert path_matches_any("anything.py", []) is None


def test_path_matches_any_skips_blank_patterns() -> None:
    assert path_matches_any("a.py", ["", "   ", "a.py"]) == "a.py"


def test_path_matches_any_normalises_backslashes_and_dot_prefix() -> None:
    assert path_matches_any("./src/a.py", ["src/*.py"]) == "src/*.py"
    assert path_matches_any("src\\a.py", ["src/*.py"]) == "src/*.py"


def test_glob_special_regex_characters_are_escaped() -> None:
    # A '+' in the path must be a literal, not a regex quantifier.
    assert glob_to_regex("src/a+b/*.py").match("src/a+b/x.py")
    assert not glob_to_regex("src/a+b/*.py").match("src/aab/x.py")


def test_nested_pyproject_does_not_shadow_the_real_root(tmp_path: Path) -> None:
    """A vendored subproject's lone pyproject.toml must not win.

    The marker walk starts at the deepest directory, so an ancestor-major scan
    would return the subproject. Marker groups are therefore applied
    group-major: every ancestor is tested against the strong groups before the
    bare-pyproject fallback is considered.
    """
    repo = _make_repo(tmp_path)
    vendored = repo / "third_party" / "vendored"
    vendored.mkdir(parents=True)
    (vendored / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert resolve_repo_root(start=vendored, env={}) == repo.resolve()


def test_bare_pyproject_still_resolves_when_no_strong_marker_exists(tmp_path: Path) -> None:
    """Fallback survives for a checkout without .git (e.g. a tarball export)."""
    root = tmp_path / "exported"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    assert resolve_repo_root(start=nested, env={}) == root.resolve()


def test_submodule_with_its_own_git_is_treated_as_its_own_root(tmp_path: Path) -> None:
    """A nested dir carrying BOTH strong markers is a real repo in its own right."""
    outer = _make_repo(tmp_path)
    inner = _make_repo(outer / "vendor" / "sub")
    assert resolve_repo_root(start=inner, env={}) == inner.resolve()
